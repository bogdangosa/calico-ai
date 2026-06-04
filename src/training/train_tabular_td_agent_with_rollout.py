import os
import math
import numpy as np
from tqdm import tqdm
from loguru import logger

from src.agents.temporal_difference.tabular_td_agent_with_rollout import TabularTDAgentWithRollout
from src.engine.environments.calico_env import CalicoEnv
from src.engine.scoring.scoring import ScoringCalculator
from src.utils.config import load_config

NUM_EPISODES = 100000
EPSILON_START = 1.0
EPSILON_END = 0.1
SAVE_FREQ = 1000
MODEL_VERSION = "v2.1"
MODEL_SAVE_PATH = f"../../agent_models/micro_calico_v2/tabular_v_table_{MODEL_VERSION}.pkl"

EPSILON_DECAY = math.exp(math.log(EPSILON_END / EPSILON_START) / NUM_EPISODES)


def train():
    config = load_config("../../config/micro_calico_settings_v2.json")
    env = CalicoEnv(config)
    scorer = ScoringCalculator(config)

    agent = TabularTDAgentWithRollout(
        config,
        learning_rate=0.2,
        discount_factor=1,
        epsilon=EPSILON_START
    )

    os.makedirs(os.path.dirname(MODEL_SAVE_PATH), exist_ok=True)

    scores = []
    logger.info(f"Starting Tabular TD Rollout training for {NUM_EPISODES} episodes...")

    pbar = tqdm(range(NUM_EPISODES))
    for episode in pbar:
        env.start_game()
        state_key = agent._get_state_key(env)

        while not env.is_game_over():
            action = agent.select_action(env)
            if action is None:
                break

            env.perform_action(action)
            next_board_snapshot = env.board_matrix.copy()
            done = env.is_game_over()

            reward = scorer.evaluate_move(env.board_matrix,env.cat_tiles) if done else 0.0

            agent.update(state_key, reward, next_board_snapshot, done)

            if not done:
                state_key = agent._get_state_key(env)

        agent.epsilon = max(EPSILON_END, agent.epsilon * EPSILON_DECAY)
        scores.append(scorer.evaluate_move(env.board_matrix,env.cat_tiles))

        if episode % 100 == 0:
            avg_score = np.mean(scores[-100:])
            pbar.set_description(
                f"Ep {episode} | Avg Score: {avg_score:.2f} | Table Size: {len(agent.v_table)} | Eps: {agent.epsilon:.2f}")

        if episode % SAVE_FREQ == 0 and episode > 0:
            agent.save(MODEL_SAVE_PATH)
            logger.debug(f"Saved V-table (size: {len(agent.v_table)}) to {MODEL_SAVE_PATH}")

    agent.save(MODEL_SAVE_PATH)
    logger.info(f"Training complete. Final V-table size: {len(agent.v_table)}")
    logger.info(f"Final model saved to {MODEL_SAVE_PATH}")


if __name__ == "__main__":
    train()