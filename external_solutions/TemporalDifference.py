import copy
import random
import numpy as np
import time
import pickle
import os

from enviroment.calico_scoring import get_total_score_on_board
from utils.constants import *


# from enviroment.calico_env import CalicoEnv # Defined inline below

# --- 1. ENVIRONMENT SETUP ---
class CalicoEnv:
    def __init__(self):
        self.tile_pool = None
        self.player_tiles = []
        self.shop_tiles = []
        self.cat_tiles = []
        self.board_matrix = []  # Initially empty list
        self.move_history = []
        self.mode = ""
        self.selected_player_tile_index = 0

    # ... [Keep standard methods: get_legal_actions, perform_action, undo_action, etc.] ...
    def get_legal_actions(self):
        legal_actions = []
        if self.mode == "placing":
            # Optimization: If the board is empty, force center placement
            empty_cells = np.argwhere(self.board_matrix == NO_TILE_VALUE)
            valid_hand_indices = [i for i, t in enumerate(self.player_tiles) if t != NO_TILE_VALUE]
            for idx in valid_hand_indices:
                for r, c in empty_cells:
                    legal_actions.append(('place', idx, r, c))

        if self.mode == "buying":
            for shop_idx, shop_tile in enumerate(self.shop_tiles):
                legal_actions.append(('buy', shop_idx, None, None))
        return legal_actions

    def perform_action(self, action):
        action_type, tile_idx, row, col = action
        record = {"action_type": action_type}
        if action_type == 'place':
            record.update({
                "row": row, "col": col, "hand_index": tile_idx,
                "hand_value": self.player_tiles[tile_idx],
                "prev_tile": self.board_matrix[row][col],
                "prev_mode": self.mode,
                "prev_selected_index": self.selected_player_tile_index
            })
            self.place_tile(row, col, tile_idx)
        elif action_type == 'buy':
            record.update({
                "shop_index": tile_idx, "prev_mode": self.mode,
                "prev_selected_index": self.selected_player_tile_index,
                "hand_value": self.player_tiles[self.selected_player_tile_index],
                "shop_snapshot": copy.deepcopy(self.shop_tiles)
            })
            self.buy_tile(tile_idx)
            record.update({"new_shop_snapshot": copy.deepcopy(self.shop_tiles)})
        self.move_history.append(record)

    def undo_action(self):
        if not self.move_history: return
        record = self.move_history.pop()
        action_type = record["action_type"]
        if action_type == "place":
            self.board_matrix[record["row"]][record["col"]] = record["prev_tile"]
            self.player_tiles[record["hand_index"]] = record["hand_value"]
            self.mode = record["prev_mode"]
            self.selected_player_tile_index = record["prev_selected_index"]
        elif action_type == "buy":
            self.shop_tiles = record["shop_snapshot"].copy()
            self.player_tiles[self.selected_player_tile_index] = record["hand_value"]
            self.mode = record["prev_mode"]
            self.selected_player_tile_index = record["prev_selected_index"]

    def start_game(self, seed=None):
        if seed:
            np.random.seed(seed)
            random.seed(seed)
        self.tile_pool = self.initiate_tile_pool()
        self.player_tiles = [self.generate_random_tile() for _ in range(PLAYER_HAND_SIZE)]
        self.shop_tiles = self.initiate_shop_tiles()
        self.cat_tiles = self.initialize_cat_tiles()
        self.board_matrix = self.initialize_inner_board()
        self.initialize_outer_board("purple")
        self.mode = "placing"
        self.selected_player_tile_index = 0

    def is_game_over(self):
        inner_board = self.board_matrix[1:BOARD_SIZE - 1, 1:BOARD_SIZE - 1]
        return not np.any(inner_board == NO_TILE_VALUE)

    def buy_tile(self, tile_index: int):
        self.mode = "placing"
        bought_tile_id = self.shop_tiles[tile_index]
        self.player_tiles[self.selected_player_tile_index] = bought_tile_id
        self.replace_tile(tile_index)
        return bought_tile_id

    def place_tile(self, row: int, col: int, selected_tile_index: int):
        selected_tile_id = self.player_tiles[selected_tile_index]
        self.board_matrix[row][col] = selected_tile_id
        self.player_tiles[selected_tile_index] = NO_TILE_VALUE
        self.selected_player_tile_index = selected_tile_index
        self.mode = "buying"

    def replace_tile(self, tile_index: int):
        self.shop_tiles = np.delete(self.shop_tiles, tile_index)
        new_tile = self.generate_random_tile()
        if new_tile is None: new_tile = 0
        self.shop_tiles = np.append(self.shop_tiles, new_tile)
        return new_tile

    def initiate_tile_pool(self):
        return np.full(TILE_COLORS * TILE_PATTERNS, NR_OF_IDENTICAL_TILES)

    def initiate_shop_tiles(self):
        return [self.generate_random_tile() for _ in range(NR_OF_TILES_IN_SHOP)]

    def initialize_cat_tiles(self):
        cat_tiles = np.arange(1, CAT_TILE_TYPES + 1)
        np.random.shuffle(cat_tiles)
        return cat_tiles

    def generate_random_tile(self):
        valid_indices = np.where(self.tile_pool > 0)[0]
        if len(valid_indices) == 0: return None
        tile_id = int(np.random.choice(valid_indices))
        self.tile_pool[tile_id] -= 1
        return tile_id

    def initialize_inner_board(self):
        board_matrix = np.full((BOARD_SIZE, BOARD_SIZE), NO_TILE_VALUE, dtype=int)
        for i, (row, col) in enumerate(OBJECTIVE_POSITIONS_ON_BOARD, start=1):
            board_matrix[row, col] = -i
        return board_matrix

    def initialize_outer_board(self, board_color):
        board_border = BOARD_BORDERS[board_color]
        idx = 0
        for c in range(BOARD_SIZE): self.board_matrix[0][c] = board_border[idx]; idx += 1
        for r in range(1, BOARD_SIZE - 1): self.board_matrix[r][BOARD_SIZE - 1] = board_border[idx]; idx += 1
        for c in range(BOARD_SIZE - 1, -1, -1): self.board_matrix[BOARD_SIZE - 1][c] = board_border[idx]; idx += 1
        for r in range(BOARD_SIZE - 2, 0, -1): self.board_matrix[r][0] = board_border[idx]; idx += 1

    # --- CNN INPUT REPRESENTATION ---
    def get_cnn_state(self):
        """
        Converts the board into a 3D tensor (H, W, Channels).
        We separate Colors and Patterns into different channels so the CNN sees relationships.
        """
        # Assuming Tile IDs 0-35 map to 6 colors x 6 patterns
        # Channel 0: Colors (0-5, or -1 for empty/obj)
        # Channel 1: Patterns (0-5, or -1 for empty/obj)
        # Channel 2: Is Occupied (Binary)
        # Channel 3: Is Objective (Binary)

        H, W = self.board_matrix.shape
        cnn_input = np.zeros((H, W, 4), dtype=float)

        for r in range(H):
            for c in range(W):
                tile_id = self.board_matrix[r][c]

                if tile_id >= 0:
                    # Decode Tile ID to Color and Pattern
                    # Assuming IDs are 0..35: color = id // 6, pattern = id % 6
                    color = tile_id // 6
                    pattern = tile_id % 6

                    cnn_input[r, c, 0] = (color + 1) / 6.0  # Normalize 0-1
                    cnn_input[r, c, 1] = (pattern + 1) / 6.0  # Normalize 0-1
                    cnn_input[r, c, 2] = 1.0  # Occupied
                elif tile_id < 0 and tile_id != NO_TILE_VALUE:
                    # Objective
                    cnn_input[r, c, 3] = 1.0

        # We flatten this for the simple Dense layer if we don't use full Conv backprop,
        # But here we will implement a basic Conv layer.
        return cnn_input


# --- 2. CONVOLUTIONAL LAYER (NumPy Implementation) ---

class Conv2DLayer:
    """
    A simple implementation of a Convolutional Layer using pure NumPy.
    Optimized for small boards/kernels to avoid heavy libraries.
    """

    def __init__(self, num_filters, filter_size, input_channels, learning_rate=0.001):
        self.num_filters = num_filters
        self.filter_size = filter_size
        self.input_channels = input_channels
        self.lr = learning_rate

        # Filters shape: (num_filters, filter_size, filter_size, input_channels)
        self.filters = np.random.randn(num_filters, filter_size, filter_size, input_channels) * 0.1
        self.bias = np.zeros(num_filters)

    def iterate_regions(self, image):
        h, w, _ = image.shape
        for i in range(h - self.filter_size + 1):
            for j in range(w - self.filter_size + 1):
                im_region = image[i:(i + self.filter_size), j:(j + self.filter_size)]
                yield i, j, im_region

    def forward(self, input_data):
        """
        Performs a forward pass of the conv layer.
        input_data: (H, W, C)
        """
        self.last_input = input_data
        h, w, _ = input_data.shape
        output_h = h - self.filter_size + 1
        output_w = w - self.filter_size + 1

        output = np.zeros((output_h, output_w, self.num_filters))

        for i, j, im_region in self.iterate_regions(input_data):
            # Apply filters: (num_filters, 3, 3, C) * (3, 3, C) -> Sum over dimensions
            for f in range(self.num_filters):
                output[i, j, f] = np.sum(im_region * self.filters[f]) + self.bias[f]

        return output

    def backward(self, d_L_d_out):
        """
        Backpropagation for Conv Layer.
        d_L_d_out: Gradient from the next layer. Shape: (Out_H, Out_W, Num_Filters)
        """
        d_L_d_filters = np.zeros(self.filters.shape)
        d_L_d_input = np.zeros(self.last_input.shape)  # Not strictly needed if first layer, but good practice

        for i, j, im_region in self.iterate_regions(self.last_input):
            for f in range(self.num_filters):
                # Gradient for filters
                d_L_d_filters[f] += d_L_d_out[i, j, f] * im_region
                # Gradient for input (omitted for speed as this is the first layer)

        # Update weights
        self.filters -= self.lr * d_L_d_filters
        self.bias -= self.lr * np.sum(d_L_d_out, axis=(0, 1))

        return None  # We don't need to backprop to the board state


class CNNValueNetwork:
    def __init__(self, board_h, board_w, channels, learning_rate=0.001):
        self.conv = Conv2DLayer(num_filters=8, filter_size=3, input_channels=channels, learning_rate=learning_rate)

        # Calculate flattened size after conv (Valid padding reduces dims by filter_size - 1)
        conv_h = board_h - 2  # 3x3 filter on 7x7 -> 5x5 output
        conv_w = board_w - 2
        self.flat_size = conv_h * conv_w * 8

        self.W1 = np.random.randn(self.flat_size, 64) * np.sqrt(2 / self.flat_size)
        self.b1 = np.zeros(64)
        self.W2 = np.random.randn(64, 1) * np.sqrt(2 / 64)
        self.b2 = np.zeros(1)
        self.lr = learning_rate

    def forward(self, X):
        # 1. Conv Layer
        out_conv = self.conv.forward(X)
        self.out_conv_shape = out_conv.shape

        # 2. Flatten
        self.flat_input = out_conv.flatten()

        # 3. Dense 1 (ReLU)
        self.z1 = np.dot(self.flat_input, self.W1) + self.b1
        self.a1 = np.maximum(0, self.z1)

        # 4. Dense 2 (Linear Output)
        self.z2 = np.dot(self.a1, self.W2) + self.b2
        self.output = self.z2[0]
        return self.output

    def backward(self, X, target):
        # Output Gradient
        output_grad = 2 * (self.output - target)

        # Dense 2 Gradients
        dW2 = np.outer(self.a1, output_grad)
        db2 = output_grad
        da1 = output_grad * self.W2.flatten()

        # Dense 1 Gradients
        dz1 = da1 * (self.z1 > 0)
        dW1 = np.outer(self.flat_input, dz1)
        db1 = dz1
        d_flat = np.dot(dz1, self.W1.T)

        # Update Dense Weights
        self.W1 -= self.lr * dW1
        self.b1 -= self.lr * db1
        self.W2 -= self.lr * dW2
        self.b2 -= self.lr * db2

        # Reshape gradient for Conv Layer
        d_conv = d_flat.reshape(self.out_conv_shape)

        # Conv Backprop
        self.conv.backward(d_conv)

        return (self.output - target) ** 2

    def save(self, filename="cnn_model.pkl"):
        with open(filename, 'wb') as f:
            pickle.dump({'conv': self.conv.filters, 'W1': self.W1, 'W2': self.W2}, f)

    def load(self, filename="cnn_model.pkl"):
        if os.path.exists(filename):
            with open(filename, 'rb') as f:
                data = pickle.load(f)
                self.conv.filters = data['conv']
                self.W1 = data['W1']
                self.W2 = data['W2']
            return True
        return False


# --- 3. TD AGENT WITH CNN ---

class TDAgent:
    def __init__(self, env, gamma=0.98, epsilon=1.0, epsilon_min=0.1, epsilon_decay=0.9995):
        self.env = env
        # Initialize CNN: 7x7 board, 4 channels
        self.vn = CNNValueNetwork(BOARD_SIZE, BOARD_SIZE, 4)
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay

    def get_best_action(self, env):
        legal_actions = env.get_legal_actions()
        if not legal_actions: return None

        best_action = None
        best_value = -float('inf')

        # Pruning: Only check neighbors if board not empty
        inner_board = env.board_matrix[1:BOARD_SIZE - 1, 1:BOARD_SIZE - 1]
        has_tiles = np.any(inner_board != NO_TILE_VALUE)
        candidates = legal_actions

        if env.mode == "placing" and has_tiles:
            candidates = []
            for act in legal_actions:
                r, c = act[2], act[3]
                neighbors = [
                    env.board_matrix[r - 1][c], env.board_matrix[r + 1][c],
                    env.board_matrix[r][c - 1], env.board_matrix[r][c + 1]
                ]
                if any(n != NO_TILE_VALUE for n in neighbors):
                    candidates.append(act)
            if not candidates: candidates = legal_actions

        for action in candidates:
            env.perform_action(action)
            state_3d = env.get_cnn_state()  # Use new CNN state
            value = self.vn.forward(state_3d)
            env.undo_action()

            if value > best_value:
                best_value = value
                best_action = action

        return best_action, best_value

    def train_episode(self):
        self.env.start_game()

        current_state = self.env.get_cnn_state()
        # Initial value estimate
        current_value = self.vn.forward(current_state)

        done = False
        total_loss = 0
        steps = 0

        while not done:
            if np.random.rand() < self.epsilon:
                legal = self.env.get_legal_actions()
                if not legal: break
                action = random.choice(legal)
                estimated_next_val = 0
            else:
                action, val = self.get_best_action(self.env)
                estimated_next_val = val

            if action is None: break

            self.env.perform_action(action)
            next_state = self.env.get_cnn_state()
            done = self.env.is_game_over()

            if done:
                actual_score = get_total_score_on_board(self.env.board_matrix, self.env.cat_tiles)
                target = actual_score
            else:
                if self.epsilon > 0:
                    target_val = self.vn.forward(next_state)
                else:
                    target_val = estimated_next_val
                target = 0 + self.gamma * target_val

            loss = self.vn.backward(current_state, target)
            total_loss += loss

            current_state = next_state
            steps += 1

        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

        return get_total_score_on_board(self.env.board_matrix, self.env.cat_tiles), total_loss / steps


if __name__ == "__main__":
    env = CalicoEnv()
    env.start_game()

    agent = TDAgent(env)

    if agent.vn.load("cnn_model.pkl"):
        print("Loaded CNN weights.")
        agent.epsilon = 0.3

    print("Starting CNN Training...")
    episodes = 2000
    scores = []

    for e in range(episodes):
        score, avg_loss = agent.train_episode()
        scores.append(score)

        if e % 10 == 0:
            avg_score = np.mean(scores[-100:])
            print(
                f"Episode {e}/{episodes} | Score: {score} | Avg Score: {avg_score:.2f} | Loss: {avg_loss:.4f} | Epsilon: {agent.epsilon:.4f}")

        if e % 100 == 0:
            agent.vn.save("cnn_model.pkl")

    agent.vn.save("cnn_model.pkl")
    print("Training Complete.")