import random
import numpy as np
from collections import defaultdict

from enviroment.calico_env import CalicoEnv
from utils.constants import *

from enviroment.calico_scoring import get_total_score_on_board


# ------------------------------
# Simple Q-learning Agent
# ------------------------------
class QLearningAgent:
    def __init__(self, action_size, lr=0.1, gamma=0.95, epsilon=0.2):
        self.Q = defaultdict(float)
        self.lr = lr
        self.gamma = gamma
        self.epsilon = epsilon
        self.action_size = action_size

    def select_action(self, state):
        """
        Epsilon-greedy action selection.
        `state` should be a 1D NumPy array (from env.get_flat_state())
        """
        state_tuple = tuple(state)  # flatten already handled by env
        if random.random() < self.epsilon:
            return random.randint(0, self.action_size - 1)
        qs = [self.Q[(state_tuple, a)] for a in range(self.action_size)]
        return int(np.argmax(qs))

    def update(self, state, action, reward, next_state, done):
        """
        Q-learning update for a given transition.
        """
        state_tuple = tuple(state)
        next_state_tuple = tuple(next_state)

        key = (state_tuple, action)
        best_next_q = 0 if done else max([self.Q[(next_state_tuple, a)] for a in range(self.action_size)])
        td_target = reward + self.gamma * best_next_q
        self.Q[key] += self.lr * (td_target - self.Q[key])

import numpy as np
import random
from collections import defaultdict

# ------------------------------
# Map action index to (hand_tile_index, row, col)
# ------------------------------
def action_to_move(action, hand_size, board_size):
    tile_index = action // (board_size * board_size)
    rem = action % (board_size * board_size)
    row = rem // board_size
    col = rem % board_size
    return tile_index, row, col

# ------------------------------
# Training function
# ------------------------------
def train_q_learning_agent(env_class, agent, num_episodes=500, max_moves=22):
    board_size = 5  # adapt if you want 7x7
    hand_size = PLAYER_HAND_SIZE
    action_size = board_size * board_size * hand_size

    for episode in range(num_episodes):
        env = env_class()
        env.start_game()
        done = False
        total_reward = 0

        for move in range(max_moves):
            state = env.get_flat_state()
            action = agent.select_action(state)
            tile_idx, row, col = action_to_move(action, hand_size, board_size)
            print(tile_idx,row, col)

            # Skip invalid actions
            if tile_idx >= len(env.player_tiles) or env.player_tiles[tile_idx] == NO_TILE_VALUE:
                continue
            if env.board_matrix[row, col] != NO_TILE_VALUE:
                continue

            # Apply the action: place tile
            env.place_tile(row, col,tile_idx)
            print(env)

            # Reward = total score after move
            reward = get_total_score_on_board(env.board_matrix, env.cat_tiles)
            total_reward += reward

            next_state = env.get_flat_state()
            done = all(t == NO_TILE_VALUE for t in env.player_tiles)
            agent.update(state, action, reward, next_state, done)

            if done:
                break

        if (episode + 1) % 50 == 0:
            print(f"Episode {episode + 1}, total_reward: {total_reward}")

    print("Training complete.")
    return agent

env_class = CalicoEnv
agent = QLearningAgent(action_size=PLAYER_HAND_SIZE * 25)
trained_agent = train_q_learning_agent(env_class, agent, num_episodes=500)
