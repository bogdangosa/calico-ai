import gymnasium as gym
from gymnasium import spaces
import numpy as np
import random

from enviroment.calico_env import CalicoEnv
from utils.constants import *
from enviroment.calico_scoring import get_total_score_on_board

class CalicoGymEnv(gym.Env):
    def __init__(self):
        super().__init__()

        # Game setup
        self.env = CalicoEnv()
        self.env.start_game()

        # Action space: place any tile on any empty cell OR buy a shop tile
        # We flatten it: BOARD_SIZE*BOARD_SIZE + NR_OF_TILES_IN_SHOP
        self.action_space = spaces.Discrete(BOARD_SIZE*BOARD_SIZE)

        # Observation space: all flattened state values (tile_pool + player + shop + cat + board)
        flat_state = self.env.get_flat_state()
        self.observation_space = spaces.Box(low=-100, high=100, shape=flat_state.shape, dtype=np.int32)

    def reset(self, seed=None, options=None):
        self.env.start_game(seed=seed)
        return self.env.get_flat_state().astype(np.int32), {}

    def step(self, action):
        # Determine if action is 'place' or 'buy'
        if action < BOARD_SIZE * BOARD_SIZE:
            row = action // BOARD_SIZE
            col = action % BOARD_SIZE
            tile_idx = next((i for i, t in enumerate(self.env.player_tiles) if t != NO_TILE_VALUE), None)
            if tile_idx is None or self.env.board_matrix[row, col] != NO_TILE_VALUE:
                reward = -1  # illegal move penalty
            else:
                self.env.place_tile(row, col, tile_idx)
                reward = get_total_score_on_board(self.env.board_matrix, self.env.cat_tiles)
        else:
            shop_idx = action - BOARD_SIZE * BOARD_SIZE
            if shop_idx >= len(self.env.shop_tiles):
                reward = -1  # illegal
            else:
                tile_idx = next((i for i, t in enumerate(self.env.player_tiles) if t == NO_TILE_VALUE), 0)
                self.env.selected_player_tile_index = tile_idx
                self.env.buy_tile(shop_idx)
                reward = 0

        terminated = self.env.is_game_over()  # game finished
        truncated = False  # optionally True if using max episode length
        info = {}
        obs = self.env.get_flat_state().astype(np.int32)

        return obs, reward, terminated, truncated, info

    def render(self):
        print(self.env)

# Example usage
if __name__ == "__main__":
    env = CalicoGymEnv()
    obs, _ = env.reset()
    env.render()

    # Random action
    action = env.action_space.sample()
    obs, reward, terminated, truncated, info = env.step(action)
    env.render()
    print("Reward:", reward, "Done:", terminated)
