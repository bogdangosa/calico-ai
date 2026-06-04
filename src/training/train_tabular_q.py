import os
import math
import numpy as np
from tqdm import tqdm
from loguru import logger

from src.agents.q_learning.tabular_q_learning_agent import TabularQLearningAgent
from src.engine.environments.calico_env import CalicoEnv
from src.engine.scoring.scoring import ScoringCalculator
from src.utils.config import load_config

# Hyperparameters
NUM_EPISODES = 300000
EPSILON_START = 1.0
EPSILON_END = 0.1
SAVE_FREQ = 1000
MODEL_VERSION = "v2.0"
MODEL_SAVE_PATH = f"../../agent_models/micro_calico_v2/tabular_q_table_{MODEL_VERSION}.pkl"

EPSILON_DECAY = math.exp(math.log(EPSILON_END / EPSILON_START) / NUM_EPISODES)

def train():
    # Setup and Initialization
    config = load_config("../../config/micro_calico_settings_v2.json")
    env = CalicoEnv(config)
    scorer = ScoringCalculator(config)

    agent = TabularQLearningAgent(
        config,
        learning_rate=0.1,
        discount_factor=0.95,
        epsilon=EPSILON_START
    )

    os.makedirs(os.path.dirname(MODEL_SAVE_PATH), exist_ok=True)

    scores = []
    logger.info(f"Starting Tabular Q-Learning training for {NUM_EPISODES} episodes...")

    # Episode Execution Loop
    pbar = tqdm(range(NUM_EPISODES))
    for episode in pbar:
        env.start_game()
        state_key = agent._get_state_key(env)

        while not env.is_game_over():
            # Request an action choice
            action = agent.select_action(env)
            if action is None:
                break

            # Resolve action key BEFORE performing action
            action_key = agent.get_action_key(action, env)

            # Execute that chosen action directly inside the environment engine
            env.perform_action(action)

            # Capture snapshots for the update
            next_state_key = agent._get_state_key(env)
            next_legal_actions = env.get_legal_actions()
            done = env.is_game_over()

            # Reward Assignment
            if done:
                reward = scorer.evaluate_move(env.board_matrix, env.cat_tiles)
            else:
                reward = 0.0

            # Update Q-table
            agent.update(state_key, action_key, reward, env, next_state_key, next_legal_actions, done)

            # Transition reference
            state_key = next_state_key

        # Decay epsilon at the conclusion of each game
        agent.epsilon = max(EPSILON_END, agent.epsilon * EPSILON_DECAY)

        final_score = scorer.evaluate_move(env.board_matrix, env.cat_tiles)
        scores.append(final_score)

        # Logging and Persistence
        if episode % 100 == 0:
            avg_score = np.mean(scores[-100:])
            pbar.set_description(f"Ep {episode} | Avg Score: {avg_score:.2f} | Q-Table Size: {len(agent.q_table)} | Eps: {agent.epsilon:.2f}")

        if episode % SAVE_FREQ == 0 and episode > 0:
            agent.save(MODEL_SAVE_PATH)
            logger.debug(f"Saved Q-table (size: {len(agent.q_table)}) to {MODEL_SAVE_PATH}")

    # Final Save
    agent.save(MODEL_SAVE_PATH)
    logger.info(f"Training complete. Final Q-table size: {len(agent.q_table)}")
    logger.info(f"Final model saved to {MODEL_SAVE_PATH}")

if __name__ == "__main__":
    train()
