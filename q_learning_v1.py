import numpy as np
import random
from collections import defaultdict

from utils.constants import PLAYER_HAND_SIZE, NR_OF_TILES_IN_SHOP, NO_TILE_VALUE


# ------------------------------
# Example stub for your score function
# Replace with your actual version.
# ------------------------------
def compute_score(board):
    return np.sum(board != 0)  # dummy: score = number of placed tiles

# ------------------------------
# Environment skeleton
# ------------------------------
import numpy as np
import random


class CalicoEnv1:
    def __init__(self, scoring_function):
        self.board_size = 5
        self.num_colors = 6  # example value
        self.num_patterns = 6
        self.tiles_per_type = 3

        self.scoring_function = scoring_function

        # Pool of available tiles (count of each type)
        self.tile_pool = self.initiate_tile_pool()

        # State
        self.board = np.full((self.board_size, self.board_size), NO_TILE_VALUE)  # -1 = empty
        self.player_tiles = []
        self.shop_tiles = []

        self.mode = "placing"  # alternates between "placing" and "buying"
        self.score = 0
        self.done = False

        self.reset()

    def initiate_tile_pool(self):
        return np.full(self.num_colors * self.num_patterns, self.tiles_per_type)

    def generate_random_tile(self):
        """Randomly pick a tile ID from pool."""
        valid_indices = np.where(self.tile_pool > 0)[0]
        if len(valid_indices) == 0:
            return None
        tile_id = int(np.random.choice(valid_indices))
        self.tile_pool[tile_id] -= 1
        return tile_id

    def reset(self):
        self.board[:] = -1
        self.tile_pool[:] = self.tiles_per_type
        self.player_tiles = [self.generate_random_tile() for _ in range(PLAYER_HAND_SIZE)]
        self.shop_tiles = [self.generate_random_tile() for _ in range(NR_OF_TILES_IN_SHOP)]
        self.mode = "placing"
        self.score = 0
        self.done = False
        return self.get_state()

    def get_state(self):
        """
        Returns a flattened or tensor version of the state.
        For now: (2, 5, 5) tensor
        [0,:,:] = board tiles, [1,:,:] = normalized scores / empty markers
        """
        board_layer = np.array(self.board, dtype=np.float32) / (self.num_colors * self.num_patterns)
        hand_layer = np.zeros((self.board_size, self.board_size), dtype=np.float32)
        return np.stack([board_layer, hand_layer], axis=0)

    def step(self, action):
        """
        action: int
          If placing: 0–(25×2−1)
          If buying: 0–(2×3−1)
        """
        reward = 0

        if self.mode == "placing":
            tile_index = action // 25  # 0 or 1 (which tile in hand)
            cell_index = action % 25
            row, col = divmod(cell_index, 5)

            # Invalid move
            if self.board[row, col] != -1:
                return self.get_state(), -10, False, {}

            tile_id = self.player_tiles[tile_index]
            self.board[row, col] = tile_id
            self.player_tiles[tile_index] = -1

            new_score = self.scoring_function(self.board)
            reward = new_score - self.score
            self.score = new_score

            self.mode = "buying"

        elif self.mode == "buying":
            hand_index = action // 3
            shop_index = action % 3

            bought_tile = self.shop_tiles[shop_index]
            self.player_tiles[hand_index] = bought_tile
            self.shop_tiles[shop_index] = self.generate_random_tile()
            self.mode = "placing"

        # Check for end condition (board full)
        if np.all(self.board != -1):
            self.done = True

        return self.get_state(), reward, self.done, {}

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
        if random.random() < self.epsilon:
            return random.randint(0, self.action_size - 1)
        qs = [self.Q[(tuple(state.flatten()), a)] for a in range(self.action_size)]
        return int(np.argmax(qs))

    def update(self, state, action, reward, next_state, done):
        key = (tuple(state.flatten()), action)
        next_qs = [self.Q[(tuple(next_state.flatten()), a)] for a in range(self.action_size)]
        best_next_q = max(next_qs) if not done else 0
        td_target = reward + self.gamma * best_next_q
        self.Q[key] += self.lr * (td_target - self.Q[key])
