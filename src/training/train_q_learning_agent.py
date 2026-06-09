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
from src.engine.scoring.scoring import ScoringCalculator
from src.training.replay_buffer import ReplayBuffer
from src.engine.simulator import run_simulation
from src.training.replay_buffer_actions import ReplayBufferActions
from src.utils.config import load_config
from src.utils.neural_network_tracker import NeuralTensorTracker

GAMMA = 0.98
BATCH_SIZE = 64
LR = 0.0001
REPLAY_CAPACITY = 100000
TARGET_UPDATE_FREQ = 10
EPSILON_START = 1.0
EPSILON_END = 0.1
NUM_EPISODES = 500000
SAVE_FREQ = 20000

# Logging Configuration
METRICS_LOG_FREQ = 100  # How often to log scalars (loss, score, etc.)
WEIGHTS_LOG_FREQ = 10000 # Increased freq
ACTIVATIONS_LOG_FREQ = 10000 # Increased freq
ENABLE_TENSOR_TRACKER = False # Toggle to completely disable the tracker

EPSILON_DECAY = math.exp(math.log(EPSILON_END / EPSILON_START) / NUM_EPISODES)

MODEL_VERSION = "v1.1"
MODEL_SAVE_PATH = f"../../agent_models/full_calico/baseline_q_learning_agent_{MODEL_VERSION}.pth"
CURRENT_TIME = datetime.now().strftime("%Y.%m.%d-%H:%M:%S")
LOG_DIR = f"../../outputs/logs/full_calico_training/{MODEL_VERSION}-{CURRENT_TIME}"

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
    scorer = ScoringCalculator(config)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Training on device: {device}")

    agent = BaselineQLearningAgent(config, device=device)
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

    tracker = None
    if ENABLE_TENSOR_TRACKER:
        tracker = NeuralTensorTracker(
            log_dir=LOG_DIR,
            activations_log_freq=ACTIVATIONS_LOG_FREQ,
            weights_log_freq=WEIGHTS_LOG_FREQ
        )
        tracker.writer = writer
        tracker.attach(agent.model)

    memory = ReplayBufferActions(REPLAY_CAPACITY)

    scores = []
    losses = []

    pbar = tqdm(range(NUM_EPISODES))
    for episode in pbar:
        env.start_game()
        state_tensor, flat_tensor = agent.get_state_tensors(env)

        prev_score = 0

        while not env.is_game_over():
            action = agent.select_action(env)
            if action is None:
                break

            action_idx = agent._get_action_index(action)

            env.perform_action(action)
            next_state_tensor, next_flat_tensor = agent.get_state_tensors(env)
            next_mask = agent.get_action_mask(env).cpu().numpy()

            current_score = scorer.evaluate_move(env.board_matrix, env.cat_tiles)
            reward = current_score - prev_score
            prev_score = current_score

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

            if len(memory) > BATCH_SIZE:
                b_state, b_flat, b_action, b_reward, b_next_state, b_next_flat, b_next_mask, b_done = memory.sample(BATCH_SIZE)

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
                    # Mask invalid actions in the next state
                    next_q_values[~b_next_mask] = float('-inf')
                    max_next_q = next_q_values.max(1)[0].unsqueeze(1)
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

        scores.append(prev_score)

        if episode % METRICS_LOG_FREQ == 0:
            avg_score = np.mean(scores[-METRICS_LOG_FREQ:]) if scores else 0
            avg_loss = np.mean(losses[-METRICS_LOG_FREQ*5:]) if losses else 0
            current_lr = optimizer.param_groups[0]['lr']

            writer.add_scalar("Metrics/Average_Score", avg_score, episode)
            writer.add_scalar("Metrics/Loss", avg_loss, episode)
            writer.add_scalar("Hyperparameters/Epsilon", agent.epsilon, episode)
            writer.add_scalar("Hyperparameters/Learning_Rate", current_lr, episode)

            if tracker:
                tracker.log_weights(agent.model, episode)

            pbar.set_description(
                f"Ep {episode} | Avg Score: {avg_score:.1f} | Loss: {avg_loss:.4f} | LR: {current_lr:.6f} | Eps: {agent.epsilon:.2f}")

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

    if tracker:
        tracker.close()


if __name__ == "__main__":
    train()
