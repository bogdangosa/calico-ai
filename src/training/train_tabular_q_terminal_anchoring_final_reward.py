import os
import math
import numpy as np
from tqdm import tqdm
from loguru import logger
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime

from src.agents.q_learning.tabular_q_learning_agent_terminal_anchoring import TabularQLearningAgentTerminalAnchoring
from src.engine.environments.calico_env import CalicoEnv
from src.engine.scoring.scoring import ScoringCalculator
from src.engine.scoring.potential_scoring import PotentialScoringCalculator
from src.utils.config import load_config

# Hyperparameters
NUM_EPISODES = 500000
EPSILON_START = 1.0
EPSILON_END = 0.1
SAVE_FREQ = 10000
MODEL_VERSION = "v1.1_anchored_potential"
MODEL_SAVE_PATH = f"../../agent_models/micro_calico_v2/tabular_q_anchored_{MODEL_VERSION}.pkl"
CURRENT_TIME = datetime.now().strftime("%Y.%m.%d-%H:%M:%S")
LOG_DIR = f"../../outputs/logs/tabular_q_anchored_training/{MODEL_VERSION}-{CURRENT_TIME}"

EPSILON_DECAY = math.exp(math.log(EPSILON_END / EPSILON_START) / NUM_EPISODES)

def train():
    config = load_config("../../config/micro_calico_settings_v2.json")
    writer = SummaryWriter(log_dir=LOG_DIR)
    
    env = CalicoEnv(config)
    actual_scorer = ScoringCalculator(config)
    potential_scorer = PotentialScoringCalculator(config)

    agent = TabularQLearningAgentTerminalAnchoring(
        config,
        learning_rate=0.1,
        discount_factor=0.95,
        epsilon=EPSILON_START
    )

    os.makedirs(os.path.dirname(MODEL_SAVE_PATH), exist_ok=True)

    scores = []
    logger.info(f"Starting Tabular Q-Learning with Terminal Anchoring for {NUM_EPISODES} episodes...")

    pbar = tqdm(range(NUM_EPISODES))
    for episode in pbar:
        env.start_game()
        state_key = agent._get_state_key(env)
        prev_score = 0

        while not env.is_game_over():
            empty_slots = np.sum(env.board_matrix == config.board.no_tile_value)

            if empty_slots == 1:
                legal_actions = env.get_legal_actions()
                best_action = None
                max_score = -1

                for act in legal_actions:
                    action_key = agent.get_action_key(act, env)
                    env.enable_history()
                    env.perform_action(act)
                    
                    final_score, *_ = potential_scorer.get_total_detailed_score(env.board_matrix, env.cat_tiles)
                    terminal_reward = final_score - prev_score
                    
                    # DIRECT UPDATE to the Q-table for ALL terminal actions (Perfect Grounding)
                    agent.update(state_key, action_key, terminal_reward, env, None, [], True)
                    
                    if final_score > max_score:
                        max_score = final_score
                        best_action = act
                    
                    env.undo_action()

                # Officially take the best action
                env.perform_action(best_action)
                prev_score = max_score
                break

            # Regular step
            action = agent.select_action(env)
            if action is None: break

            action_key = agent.get_action_key(action, env)
            env.perform_action(action)

            next_state_key = agent._get_state_key(env)
            next_legal_actions = env.get_legal_actions()
            done = env.is_game_over()

            current_score = potential_scorer.evaluate_move(env.board_matrix, env.cat_tiles)
            reward = current_score - prev_score
            
            agent.update(state_key, action_key, reward, env, next_state_key, next_legal_actions, done)

            state_key = next_state_key
            prev_score = current_score

        agent.epsilon = max(EPSILON_END, agent.epsilon * EPSILON_DECAY)
        scores.append(prev_score)

        if episode % 1000 == 0:
            avg_score = np.mean(scores[-1000:]) if scores else 0
            writer.add_scalar("Metrics/Average_Score", avg_score, episode)
            writer.add_scalar("Metrics/Table_Size", len(agent.q_table), episode)
            pbar.set_description(f"Ep {episode} | Avg Score: {avg_score:.2f} | Q-Size: {len(agent.q_table)}")

        if episode % SAVE_FREQ == 0 and episode > 0:
            agent.save(MODEL_SAVE_PATH)

    agent.save(MODEL_SAVE_PATH)
    logger.info(f"Training complete. Final Q-table size: {len(agent.q_table)}")
    writer.close()

if __name__ == "__main__":
    train()
