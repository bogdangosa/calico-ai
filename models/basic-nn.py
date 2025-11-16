import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import random

from models.random_model import get_random_tile_from_hand
from random_model import get_random_tile_to_buy
from enviroment.calico_env import CalicoEnv
from utils.constants import *

from enviroment.calico_scoring import get_total_score_on_board

class TilePlacementNet(nn.Module):
    def __init__(self, state_size, action_size, hidden_size=128):
        super(TilePlacementNet, self).__init__()
        self.fc1 = nn.Linear(state_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, hidden_size)
        self.fc4 = nn.Linear(hidden_size, action_size)

    def forward(self, x, mask=None):
        # x: [batch_size, state_size]
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        out = self.fc4(x)
        if mask is not None:
            out = out + (mask - 1) * 1e6  # mask out invalid moves
        return out

def board_to_mask(board_matrix):
    """
    Converts a 5x5 board matrix into a mask where:
      - cells with value 99 → 1 (legal/empty)
      - all other cells → 0 (occupied)
    """
    board = np.array(board_matrix)
    mask = (board == NO_TILE_VALUE).astype(np.float32)
    return mask.flatten()

def play_turn(env, trained_model,epsilon=0.1):
    # Place tile
    tile_placed = get_random_tile_from_hand()
    mask = board_to_mask(env.board_matrix)
    flat_state = env.get_flat_state()
    flat_state = np.clip(flat_state, -10, 100) / 100.0

    # Convert to torch tensor, add batch dimension, and float32
    state_tensor = torch.tensor(flat_state, dtype=torch.float32).unsqueeze(0)  # shape [1, state_size]
    mask_tensor = torch.tensor(mask, dtype=torch.float32).unsqueeze(0)  # shape [1, BOARD_SIZE*BOARD_SIZE] if needed

    # Forward pass
    position_placed = trained_model.forward(state_tensor, mask_tensor)

    # --- ε-greedy exploration ---
    if random.random() < epsilon:
        # 🎲 Explore: random valid position
        valid_positions = np.where(mask == 1)[0]
        if len(valid_positions) == 0:
            return None
        position_placed = int(np.random.choice(valid_positions))
    else:
        # 🧠 Exploit: choose best move
        with torch.no_grad():
            q_values = trained_model.forward(state_tensor, mask_tensor)
        position_placed = int(q_values.squeeze(0).argmax().item())

    row = position_placed // BOARD_SIZE
    column = position_placed % BOARD_SIZE
    env.place_tile(row, column, tile_placed)

    # Buy tile
    tile_bought_id = get_random_tile_to_buy()
    env.buy_tile(tile_bought_id)
    return position_placed

def play_game(trained_model):
    env = CalicoEnv()
    env.start_game()
    print(env)
    while not env.is_game_over():
        play_turn(env,trained_model)
    print(env)
    print(get_total_score_on_board(env.board_matrix,env.cat_tiles))

def train_model(model, env_class, episodes=5000, lr=0.01, gamma=0.99, epsilon=1, epsilon_decay=0.9995, min_epsilon=0.1):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    all_rewards = []

    for ep in range(episodes):
        env = env_class()
        env.start_game()
        total_reward = 0
        new_score = 0
        while not env.is_game_over():

            flat_state = torch.tensor(env.get_flat_state(), dtype=torch.float32).unsqueeze(0)
            flat_state = np.clip(flat_state, -10, 100) / 100.0
            mask = board_to_mask(env.board_matrix)
            mask_tensor = torch.tensor(mask, dtype=torch.float32).unsqueeze(0)

            action = play_turn(env, model,epsilon)

            # Reward = delta score
            old_score = new_score
            new_score = get_total_score_on_board(env.board_matrix, env.cat_tiles)
            if new_score == old_score:
                reward = -0.1  # discourage doing nothing
            else:
                reward = (new_score - old_score) * 10  # reward progress
            total_reward = new_score

            # --- Train step ---
            optimizer.zero_grad()
            q_pred = model(flat_state, mask_tensor)[0, action]
            # Estimate next max
            if env.is_game_over():
                q_target = torch.tensor(reward, dtype=torch.float32)
            else:
                next_state = torch.tensor(env.get_flat_state(), dtype=torch.float32).unsqueeze(0)
                next_state = np.clip(next_state, -10, 100) / 100.0
                next_mask = (env.board_matrix.flatten() == NO_TILE_VALUE).astype(np.float32)
                next_mask_tensor = torch.tensor(next_mask, dtype=torch.float32).unsqueeze(0)
                with torch.no_grad():
                    q_target = reward + gamma * model(next_state, next_mask_tensor).max()
            loss = F.mse_loss(q_pred, q_target)
            loss.backward()
            optimizer.step()

        # Decay epsilon
        epsilon = max(min_epsilon, epsilon * epsilon_decay)
        all_rewards.append(total_reward)

        if (ep+1) % 100 == 0:
            #print(env)
            #print(get_total_score_on_board(env.board_matrix, env.cat_tiles))
            print(f"Episode {ep+1}, Total Reward: {total_reward}, Epsilon: {epsilon:.3f}, Avg Last 100: {np.mean(all_rewards[-100:]):.2f}")

    return model, all_rewards

state_size = 37
action_size = BOARD_SIZE * BOARD_SIZE
model = TilePlacementNet(state_size, action_size)

trained_model, rewards = train_model(model, CalicoEnv, episodes=20000)