import math
import os
from datetime import datetime

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from tqdm import tqdm
from loguru import logger
from torch.utils.tensorboard import SummaryWriter

from src.agents.q_learning.baseline_q_learning_agent import BaselineQLearningAgent
from src.engine.environments.calico_env import CalicoEnv
from src.engine.scoring.potential_scoring import PotentialScoringCalculator
from src.engine.scoring.scoring import ScoringCalculator
from src.training.replay_buffer import ReplayBuffer
from src.engine.simulator import run_simulation
from src.training.replay_buffer_actions import ReplayBufferActions
from src.utils.config import load_config

# --- Hyperparameters ---
GAMMA = 0.98
BATCH_SIZE = 64
LR = 0.00005
REPLAY_CAPACITY = 100000
TARGET_UPDATE_FREQ = 10
EPSILON_START = 1.0  # Start from 1.0 for training from scratch
EPSILON_END = 0.05
NUM_EPISODES = 500000
SAVE_FREQ = 10000

# Logging Configuration
METRICS_LOG_FREQ = 1000
ENABLE_TENSOR_TRACKER = False

EPSILON_DECAY = math.exp(math.log(EPSILON_END / EPSILON_START) / NUM_EPISODES)

MODEL_VERSION = "v1.3"
LOAD_MODEL_PATH = None
MODEL_SAVE_PATH = f"../../agent_models/full_calico/baseline_q_terminal_anchoring_{MODEL_VERSION}.pth"
CURRENT_TIME = datetime.now().strftime("%Y.%m.%d-%H:%M:%S")
LOG_DIR = f"../../outputs/logs/full_calico_anchoring_training/{MODEL_VERSION}-{CURRENT_TIME}"

LR_SCHEDULER_STEP = 100000
LR_SCHEDULER_GAMMA = 0.5

EVAL_NUM_GAMES = 20


def evaluate_agent(env, agent, num_games=20):
    old_epsilon = agent.epsilon
    agent.epsilon = 0.0
    agent.model.eval()

    scores = run_simulation(
        env=env,
        agent=agent,
        num_games=num_games,
        progress_interval=num_games,
        plot_score_interval=False,
        save_to_dataset=False
    )

    agent.epsilon = old_epsilon
    agent.model.train()
    return np.mean(scores)


def train():
    config = load_config("../../config/calico_settings.json")
    writer = SummaryWriter(log_dir=LOG_DIR)

    env = CalicoEnv(config)
    env.enable_history()
    potential_scorer = PotentialScoringCalculator(config)
    actual_scorer = ScoringCalculator(config)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Training Baseline Q with Terminal Anchoring on: {device}")

    agent = BaselineQLearningAgent(config, model_path=LOAD_MODEL_PATH, device=device)
    agent.epsilon = EPSILON_START
    agent.model.train()

    target_net = type(agent.model)(
        input_channels=agent.encoder.total_feature_layers,
        board_size=config.board.size,
        flat_features_size=agent.encoder.flat_features_size,
        action_space_size=agent.action_space_size
    ).to(device)
    target_net.load_state_dict(agent.model.state_dict())
    target_net.eval()

    optimizer = optim.Adam(agent.model.parameters(), lr=LR)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=LR_SCHEDULER_STEP, gamma=LR_SCHEDULER_GAMMA)
    criterion = nn.SmoothL1Loss()

    memory = ReplayBufferActions(REPLAY_CAPACITY)

    scores = []
    losses = []

    pbar = tqdm(range(NUM_EPISODES))
    for episode in pbar:
        env.start_game()
        state_tensor, flat_tensor = agent.get_state_tensors(env)

        prev_potential_score = 0

        while not env.is_game_over():
            # Check if this is the last placement (exactly 1 slot left)
            empty_slots = np.sum(env.board_matrix == config.board.no_tile_value)

            if empty_slots == 1:
                # --- TERMINAL ANCHORING LOGIC ---
                legal_actions = env.get_legal_actions()
                state_np = state_tensor.squeeze(0).cpu().numpy()
                flat_np = flat_tensor.squeeze(0).cpu().numpy()

                best_action = None
                max_score = -1

                for act in legal_actions:
                    env.perform_action(act)
                    # Get actual final score for this action
                    final_score, *_ = actual_scorer.get_total_detailed_score(env.board_matrix, env.cat_tiles)

                    # Reward is (Final Actual Score - Previous Potential Score)
                    terminal_reward = final_score - prev_potential_score

                    # Store this transition as grounded truth
                    act_idx = agent._get_action_index(act)

                    # For terminal state, next tensors don't matter much but we provide dummy ones
                    memory.push(
                        state_np, flat_np, act_idx, terminal_reward,
                        state_np, flat_np, np.zeros(agent.action_space_size, dtype=bool), True
                    )

                    if final_score > max_score:
                        max_score = final_score
                        best_action = act

                    env.undo_action()

                # Perform the best terminal action to actually end the game
                env.perform_action(best_action)
                prev_potential_score = max_score  # Update for final logging
                break

            # Regular step
            action = agent.select_action(env)
            if action is None: break

            action_idx = agent._get_action_index(action)

            env.perform_action(action)
            next_state_tensor, next_flat_tensor = agent.get_state_tensors(env)
            next_mask = agent.get_action_mask(env).cpu().numpy()

            current_potential = potential_scorer.evaluate_move(env.board_matrix, env.cat_tiles)
            reward = current_potential - prev_potential_score
            prev_potential_score = current_potential

            done = env.is_game_over()

            memory.push(
                state_tensor.squeeze(0).cpu().numpy(),
                flat_tensor.squeeze(0).cpu().numpy(),
                action_idx,
                reward,
                next_state_tensor.squeeze(0).cpu().numpy(),
                next_flat_tensor.squeeze(0).cpu().numpy(),
                next_mask,
                done
            )

            state_tensor = next_state_tensor
            flat_tensor = next_flat_tensor

            # Training step
            if len(memory) > BATCH_SIZE:
                b_state, b_flat, b_action, b_reward, b_next_state, b_next_flat, b_next_mask, b_done = memory.sample(
                    BATCH_SIZE)

                b_state = torch.from_numpy(b_state).to(device).float()
                b_flat = torch.from_numpy(b_flat).to(device).float()
                b_action = torch.from_numpy(b_action).to(device).long().unsqueeze(1)
                b_reward = torch.from_numpy(b_reward).to(device).float().unsqueeze(1)
                b_next_state = torch.from_numpy(b_next_state).to(device).float()
                b_next_flat = torch.from_numpy(b_next_flat).to(device).float()
                b_next_mask = torch.from_numpy(b_next_mask).to(device)
                b_done = torch.from_numpy(b_done).to(device).float().unsqueeze(1)

                current_q = agent.model(b_state, b_flat).gather(1, b_action)

                with torch.no_grad():
                    next_q_values = target_net(b_next_state, b_next_flat)
                    next_q_values[~b_next_mask] = float('-inf')
                    max_next_q = next_q_values.max(1)[0].unsqueeze(1)

                    # Ensure max_next_q is 0 for terminal states to avoid 0 * -inf = NaN
                    max_next_q[b_done.bool()] = 0.0
                    target_q = b_reward + (1.0 - b_done) * GAMMA * max_next_q

                loss = criterion(current_q, target_q)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                losses.append(loss.item())

        if episode % TARGET_UPDATE_FREQ == 0:
            target_net.load_state_dict(agent.model.state_dict())

        agent.epsilon = max(EPSILON_END, agent.epsilon * EPSILON_DECAY)
        scheduler.step()

        scores.append(prev_potential_score)

        if episode % METRICS_LOG_FREQ == 0:
            avg_score = np.mean(scores[-METRICS_LOG_FREQ:]) if scores else 0
            avg_loss = np.mean(losses[-METRICS_LOG_FREQ * 5:]) if losses else 0
            current_lr = optimizer.param_groups[0]['lr']

            writer.add_scalar("Metrics/Average_Score", avg_score, episode)
            writer.add_scalar("Metrics/Loss", avg_loss, episode)
            writer.add_scalar("Hyperparameters/Epsilon", agent.epsilon, episode)
            writer.add_scalar("Hyperparameters/Learning_Rate", current_lr, episode)

            pbar.set_description(
                f"Ep {episode} | Avg Score: {avg_score:.1f} | Loss: {avg_loss:.4f} | Eps: {agent.epsilon:.2f}")

        if episode % SAVE_FREQ == 0 or episode == NUM_EPISODES - 1:
            os.makedirs(os.path.dirname(MODEL_SAVE_PATH), exist_ok=True)
            agent.model.save(MODEL_SAVE_PATH)

            logger.info(f"Evaluating agent at episode {episode}...")
            eval_score = evaluate_agent(env, agent, EVAL_NUM_GAMES)
            writer.add_scalar("Metrics/Evaluation_Score", eval_score, episode)
            logger.info(f"Evaluation Average Score: {eval_score:.2f}")
            agent.model.train()

    logger.info("Training complete.")
    agent.model.save(MODEL_SAVE_PATH)
    logger.info(f"Final model saved to {MODEL_SAVE_PATH}")
    writer.close()


if __name__ == "__main__":
    train()
