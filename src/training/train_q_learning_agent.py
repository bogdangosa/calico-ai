import os
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from tqdm import tqdm
from loguru import logger
from torch.utils.tensorboard import SummaryWriter

from src.engine.environments.calico_env import CalicoEnv
from src.agents.q_learning.q_learning_agent import QLearningAgent
from src.training.replay_buffer import ReplayBuffer
from src.engine.scoring.scoring import ScoringCalculator
from src.utils.config import load_config

# Hyperparameters
GAMMA = 0.95
BATCH_SIZE = 64
LR = 0.0001
REPLAY_CAPACITY = 50000
TARGET_UPDATE_FREQ = 10
EPSILON_START = 1.0
EPSILON_END = 0.1
EPSILON_DECAY = 0.99994
NUM_EPISODES = 50000
SAVE_FREQ = 500
MODEL_SAVE_PATH = "agent_models/micro_calico/q_learning_agent_v1.0.pth"
LOG_DIR = "outputs/logs/q_learning_training"

def train():
    # Load config
    config = load_config("config/micro_calico_settings.json")
    
    # Initialize TensorBoard writer
    writer = SummaryWriter(log_dir=LOG_DIR)
    
    # Initialize environment and scoring
    env = CalicoEnv(config)
    env.enable_history()
    scorer = ScoringCalculator(config)
    
    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Training on device: {device}")
    
    # Initialize Agent and Networks
    agent = QLearningAgent(config, device=device)
    agent.epsilon = EPSILON_START
    
    # Target Network
    target_net = type(agent.model)(
        input_channels=agent.encoder.total_feature_layers,
        board_size=config.board.size
    ).to(device)
    target_net.load_state_dict(agent.model.state_dict())
    target_net.eval()
    
    # Optimizer and Loss
    optimizer = optim.Adam(agent.model.parameters(), lr=LR)
    criterion = nn.MSELoss()
    
    # Replay Buffer
    memory = ReplayBuffer(REPLAY_CAPACITY)
    
    # Training Loop
    scores = []
    losses = []
    
    pbar = tqdm(range(NUM_EPISODES))
    for episode in pbar:
        env.start_game()
        state_tensor = agent.get_state_tensor(env)
        
        total_episode_reward = 0
        prev_score = 0
        
        while not env.is_game_over():
            # Select action
            action = agent.select_action(env)
            if action is None:
                break
                
            # Perform action
            env.perform_action(action)
            next_state_tensor = agent.get_state_tensor(env)
            
            # Calculate reward
            current_score = scorer.evaluate_move(env.board_matrix, env.cat_tiles)
            reward = (current_score - prev_score)
            prev_score = current_score
            
            done = env.is_game_over()
            
            # Store in replay buffer
            memory.push(state_tensor.cpu().numpy(), reward, next_state_tensor.cpu().numpy(), done)
            
            state_tensor = next_state_tensor
            
            # Optimization step
            if len(memory) > BATCH_SIZE:
                b_state, b_reward, b_next_state, b_done = memory.sample(BATCH_SIZE)
                
                b_state = torch.from_numpy(b_state).to(device)
                b_reward = torch.from_numpy(b_reward).to(device)
                b_next_state = torch.from_numpy(b_next_state).to(device)
                b_done = torch.from_numpy(b_done).to(device)
                
                # Compute current Q values
                current_q = agent.model(b_state).squeeze()
                
                # Compute target Q values
                with torch.no_grad():
                    next_q = target_net(b_next_state).squeeze()
                    target_q = b_reward + (1 - b_done) * GAMMA * next_q
                
                loss = criterion(current_q, target_q)
                
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                losses.append(loss.item())
        
        # Update target network
        if episode % TARGET_UPDATE_FREQ == 0:
            target_net.load_state_dict(agent.model.state_dict())
            
        # Decay epsilon
        agent.epsilon = max(EPSILON_END, agent.epsilon * EPSILON_DECAY)
        
        scores.append(prev_score)
        
        # Logging
        if episode % 10 == 0:
            avg_score = np.mean(scores[-10:])
            avg_loss = np.mean(losses[-50:]) if losses else 0
            
            writer.add_scalar("Metrics/Average_Score", avg_score, episode)
            writer.add_scalar("Metrics/Loss", avg_loss, episode)
            writer.add_scalar("Hyperparameters/Epsilon", agent.epsilon, episode)
            
            pbar.set_description(f"Ep {episode} | Avg Score: {avg_score:.1f} | Loss: {avg_loss:.4f} | Eps: {agent.epsilon:.2f}")
            
        # Save model
        if episode % SAVE_FREQ == 0 or episode == NUM_EPISODES - 1:
            os.makedirs(os.path.dirname(MODEL_SAVE_PATH), exist_ok=True)
            agent.model.save(MODEL_SAVE_PATH)
            
    logger.info("Training complete.")
    agent.model.save(MODEL_SAVE_PATH)
    logger.info(f"Final model saved to {MODEL_SAVE_PATH}")
    writer.close()

if __name__ == "__main__":
    train()
