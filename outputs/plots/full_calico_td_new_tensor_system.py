import numpy as np
import random
import pickle
import os
import time
import copy
import matplotlib

try:
    import tkinter

    matplotlib.use('TkAgg')
    HEADLESS = False
except ImportError:
    print("Warning: GUI libraries not found. Switching to Headless mode.")
    matplotlib.use('Agg')
    HEADLESS = True
import matplotlib.pyplot as plt
from collections import deque

# --- IMPORT YOUR BASE ENVIRONMENT & SCORING ---
from enviroment.calico_scoring import get_total_score_on_board
from utils.constants import *

# --- CONSTANTS FOR NORMALIZATION ---
# Increased to 120 to ensure we don't artificially limit the agent's perceived potential
SCORE_MAX = 120.0


# --- 1. REIMPLEMENTED CALICO ENV (With Tensor State) ---

class CalicoEnv:
    def __init__(self):
        self.tile_pool = None
        self.player_tiles = []
        self.shop_tiles = []
        self.cat_tiles = []
        self.board_matrix = []
        self.move_history = []
        self.mode = ""
        self.selected_player_tile_index = 0

    # ... [STANDARD METHODS: get_legal_actions, perform_action, etc.] ...
    def get_legal_actions(self):
        legal_actions = []
        if self.mode == "placing":
            inner_board = self.board_matrix[1:BOARD_SIZE - 1, 1:BOARD_SIZE - 1]
            has_tiles = np.any(inner_board != NO_TILE_VALUE)

            empty_cells = []
            if not has_tiles:
                empty_cells = [(3, 3)]
                for obj_pos in OBJECTIVE_POSITIONS_ON_BOARD:
                    if self.board_matrix[obj_pos[0]][obj_pos[1]] == NO_TILE_VALUE:
                        empty_cells.append(tuple(obj_pos))
            else:
                rows, cols = np.where(self.board_matrix != NO_TILE_VALUE)
                potential_spots = set()
                for r, c in zip(rows, cols):
                    for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                        nr, nc = r + dr, c + dc
                        if 1 <= nr < BOARD_SIZE - 1 and 1 <= nc < BOARD_SIZE - 1:
                            if self.board_matrix[nr][nc] == NO_TILE_VALUE:
                                potential_spots.add((nr, nc))
                empty_cells = list(potential_spots)
                if not empty_cells:
                    er, ec = np.where(self.board_matrix[1:BOARD_SIZE - 1, 1:BOARD_SIZE - 1] == NO_TILE_VALUE)
                    empty_cells = list(zip(er + 1, ec + 1))

            for idx, tile_id in enumerate(self.player_tiles):
                if tile_id == NO_TILE_VALUE: continue
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
            random.seed(seed)
            np.random.seed(seed)
        self.tile_pool = self.initiate_tile_pool()
        self.player_tiles = [self.generate_random_tile() for _ in range(PLAYER_HAND_SIZE)]
        self.shop_tiles = self.initiate_shop_tiles()
        self.cat_tiles = self.initialize_cat_tiles()
        self.board_matrix = self.initialize_inner_board()
        self.initialize_outer_board("purple")
        self.mode = "placing"
        self.selected_player_tile_index = 0

    def is_game_over(self):
        return not np.any(self.board_matrix == NO_TILE_VALUE)

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

    def initiate_player_tiles(self):
        return [self.generate_random_tile() for _ in range(PLAYER_HAND_SIZE)]

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

    def fill_board_randomly(self):
        for i in range(BOARD_SIZE - 2):
            for j in range(BOARD_SIZE - 2):
                if [i + 1, j + 1] in OBJECTIVE_POSITIONS_ON_BOARD: continue
                self.board_matrix[i + 1][j + 1] = self.generate_random_tile()

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

    # --- TENSOR REPRESENTATION ---
    def get_board_tensor(self):
        """
        Creates a 13-channel tensor representation of the board.
        """
        H, W = self.board_matrix.shape
        channels = TILE_COLORS + TILE_PATTERNS + 1
        tensor = np.zeros((H, W, channels), dtype=float)

        for r in range(H):
            for c in range(W):
                val = self.board_matrix[r][c]

                if val >= 0:  # Tile
                    color = val // TILE_PATTERNS
                    pattern = val % TILE_PATTERNS

                    if color < TILE_COLORS: tensor[r, c, color] = 1.0
                    if pattern < TILE_PATTERNS: tensor[r, c, TILE_COLORS + pattern] = 1.0

                elif val < 0 and val != NO_TILE_VALUE:  # Objective
                    tensor[r, c, 12] = 1.0

        return tensor

    def get_cnn_state(self):
        return self.get_board_tensor()

    def get_flat_state(self):
        return self.get_board_tensor().flatten()


# ==============================================================================
# PART 2: VISUALIZATION HELPER
# ==============================================================================
class TrainingVisualizer:
    def __init__(self):
        self.plot_ready = False
        self.episodes, self.scores, self.losses = [], [], []
        self.preds, self.acts = [], []

        try:
            if not HEADLESS: plt.ion()
            self.fig, self.axs = plt.subplots(2, 2, figsize=(12, 8))
            if not HEADLESS:
                self.fig.canvas.manager.set_window_title('Full Calico TD Training')
                plt.tight_layout()
                plt.show(block=False)
            self.plot_ready = True

            self.ax_trends = self.axs[0, 0]
            self.ax_trends_loss = self.ax_trends.twinx()
            self.ax_acc = self.axs[0, 1]
            self.ax_filt = self.axs[1, 0]
            self.ax_stats = self.axs[1, 1]

            self.line_score, = self.ax_trends.plot([], [], 'b-', label='Avg Score')
            self.line_loss, = self.ax_trends_loss.plot([], [], 'r-', alpha=0.3, label='Loss')
            self.scat = self.ax_acc.scatter([], [], c='purple', alpha=0.6)
            self.line_ideal, = self.ax_acc.plot([], [], 'k--')
            self.img_obj = None
            self.text_obj = self.ax_stats.text(0.05, 0.5, "Init...", fontsize=11, family='monospace')

        except Exception as e:
            print(f"Viz Init Error: {e}")

    def update(self, episode, score, avg_score, loss, pred, act, filters):
        if not self.plot_ready: return
        try:
            self.episodes.append(episode)
            self.scores.append(avg_score)
            self.losses.append(loss)
            self.preds.append(pred)
            self.acts.append(act)

            if len(self.episodes) > 200:
                self.episodes.pop(0);
                self.scores.pop(0);
                self.losses.pop(0)
                self.preds.pop(0);
                self.acts.pop(0)

            self.line_score.set_data(self.episodes, self.scores)
            self.line_loss.set_data(self.episodes, self.losses)
            self.ax_trends.relim();
            self.ax_trends.autoscale_view()
            self.ax_trends_loss.relim();
            self.ax_trends_loss.autoscale_view()
            self.ax_trends.set_title(f"Learning (Ep {episode})")

            self.scat.set_offsets(np.c_[self.preds, self.acts])
            if self.preds:
                mn, mx = min(min(self.preds), min(self.acts)), max(max(self.preds), max(self.acts))
                if mn == mx: mx += 1
                self.line_ideal.set_data([mn, mx], [mn, mx])
                self.ax_acc.set_xlim(mn, mx)
                self.ax_acc.set_ylim(mn, mx)

            if filters is not None:
                n = min(4, filters.shape[0])
                imgs = []
                fmin, fmax = filters.min(), filters.max()
                for i in range(n):
                    w = filters[i, :, :, 0]
                    if fmax - fmin > 1e-5: w = (w - fmin) / (fmax - fmin)
                    imgs.append(w)
                if imgs:
                    combined = np.hstack(imgs)
                    if self.img_obj is None:
                        self.img_obj = self.ax_filt.imshow(combined, cmap='viridis')
                        self.ax_filt.axis('off')
                    else:
                        self.img_obj.set_data(combined)

            stats = (f"Ep: {episode}\nScore: {score}\nAvg(50): {avg_score:.2f}\nLoss: {loss:.4f}\n"
                     f"Pred: {pred:.1f} | Act: {act:.1f}")
            self.text_obj.set_text(stats)

            if HEADLESS:
                plt.savefig("full_calico_dashboard.png")
            else:
                self.fig.canvas.flush_events()
        except Exception as e:
            print(f"Viz Update Error: {e}")


# ==============================================================================
# PART 3: CONVOLUTIONAL NEURAL NETWORK (Scaled & Normalized)
# ==============================================================================

class Conv2DLayer:
    def __init__(self, num_filters, filter_size, input_channels, lr=0.01):
        self.num_filters = num_filters
        self.filter_size = filter_size
        self.input_channels = input_channels
        self.lr = lr
        scale = np.sqrt(2.0 / (filter_size * filter_size * input_channels))
        self.filters = np.random.randn(num_filters, filter_size, filter_size, input_channels) * scale
        self.bias = np.zeros(num_filters)

    def forward(self, X):
        self.last_input = X
        h, w, _ = X.shape
        out_h, out_w = h - self.filter_size + 1, w - self.filter_size + 1
        output = np.zeros((out_h, out_w, self.num_filters))

        for i in range(out_h):
            for j in range(out_w):
                region = X[i:i + self.filter_size, j:j + self.filter_size]
                for f in range(self.num_filters):
                    output[i, j, f] = np.sum(region * self.filters[f]) + self.bias[f]
        return output

    def backward(self, d_out):
        d_filters = np.zeros_like(self.filters)
        h, w, _ = d_out.shape
        for i in range(h):
            for j in range(w):
                region = self.last_input[i:i + self.filter_size, j:j + self.filter_size]
                for f in range(self.num_filters):
                    d_filters[f] += d_out[i, j, f] * region

        np.clip(d_filters, -0.1, 0.1, out=d_filters)
        self.filters -= self.lr * d_filters
        self.bias -= self.lr * np.sum(d_out, axis=(0, 1))
        return None


class FullCNNValueNetwork:
    def __init__(self, board_size, channels, lr=0.02):
        # 32 filters
        self.conv = Conv2DLayer(num_filters=32, filter_size=3, input_channels=channels, lr=lr)

        conv_out_dim = board_size - 2
        self.flat_size = conv_out_dim * conv_out_dim * 32

        # 128 neurons
        self.W1 = np.random.randn(self.flat_size, 128) * np.sqrt(2 / self.flat_size)
        self.b1 = np.zeros(128)

        # Output weights
        self.W2 = np.random.randn(128, 1) * np.sqrt(2 / 128)
        # Bias init
        self.b2 = np.zeros(1) + 0.25

        self.lr = lr

    def forward(self, X):
        out_conv = self.conv.forward(X)
        self.conv_shape = out_conv.shape
        self.flat = out_conv.flatten()
        self.z1 = np.dot(self.flat, self.W1) + self.b1
        self.a1 = np.maximum(0, self.z1)
        # Linear output: Value between 0 and 1 (Normalized)
        self.output = np.dot(self.a1, self.W2) + self.b2
        return self.output[0]

    def backward(self, X, target):
        # Target here is already NORMALIZED (0-1)
        diff = self.output[0] - target
        grad = 2 * diff
        grad = np.clip(grad, -1, 1)

        dW2 = np.outer(self.a1, grad)
        db2 = grad
        da1 = grad * self.W2.flatten()
        dz1 = da1 * (self.z1 > 0)
        dW1 = np.outer(self.flat, dz1)
        db1 = dz1
        dflat = np.dot(dz1, self.W1.T)

        np.clip(dW1, -1, 1, out=dW1)

        self.W1 -= self.lr * dW1
        self.b1 -= self.lr * db1
        self.W2 -= self.lr * dW2
        self.b2 -= self.lr * db2

        dconv = dflat.reshape(self.conv_shape)
        self.conv.backward(dconv)

        return diff ** 2

    def save(self, fname="full_calico.pkl"):
        with open(fname, 'wb') as f:
            pickle.dump({'c': self.conv.filters, 'cb': self.conv.bias,
                         'w1': self.W1, 'b1': self.b1, 'w2': self.W2, 'b2': self.b2,
                         'ch': self.conv.input_channels}, f)

    def load(self, fname="full_calico.pkl"):
        if os.path.exists(fname):
            try:
                with open(fname, 'rb') as f:
                    d = pickle.load(f)
                    # Check input_channels compatibility
                    if d.get('ch', 0) != self.conv.input_channels:
                        print(f"Discarding incompatible model (Channels: {d.get('ch')} vs {self.conv.input_channels})")
                        return False
                    self.conv.filters = d['c'];
                    self.conv.bias = d['cb']
                    self.W1 = d['w1'];
                    self.b1 = d['b1']
                    self.W2 = d['w2'];
                    self.b2 = d['b2']
                return True
            except:
                return False
        return False


# ==============================================================================
# PART 4: FULL TD AGENT (Normalized)
# ==============================================================================

class FullTDAgent:
    def __init__(self, env, gamma=0.95, epsilon=1.0, decay=0.997):
        self.env = env
        self.channels = TILE_COLORS + TILE_PATTERNS + 1
        self.vn = FullCNNValueNetwork(BOARD_SIZE, self.channels)
        self.gamma = gamma
        self.epsilon = epsilon
        self.min_epsilon = 0.1
        self.decay = decay
        self.memory = deque(maxlen=5000)
        self.batch_size = 32

    def get_state(self, env):
        return env.get_board_tensor()

    def get_best_action(self, env):
        legal = env.get_legal_actions()
        if not legal: return None

        best_act = None
        best_val = -float('inf')

        for act in legal:
            env.perform_action(act)
            val = self.vn.forward(self.get_state(env))
            env.undo_action()
            if val > best_val:
                best_val = val
                best_act = act
        return best_act, best_val

    def replay(self):
        if len(self.memory) < self.batch_size: return 0
        batch = random.sample(self.memory, self.batch_size)
        loss_sum = 0
        for s, a, r, ns, done in batch:
            # Normalize Rewards for training
            r_norm = r / SCORE_MAX

            if done:
                target = r_norm
            else:
                target = r_norm + self.gamma * self.vn.forward(ns)

            loss_sum += self.vn.backward(s, target)
        return loss_sum / len(batch)

    def train_episode(self):
        self.env.start_game()
        curr_state = self.get_state(self.env)

        prev_score = get_total_score_on_board(self.env.board_matrix, self.env.cat_tiles)

        init_pred_norm = self.vn.forward(curr_state)
        init_pred_real = init_pred_norm * SCORE_MAX

        done = False
        loss_sum = 0
        steps = 0

        while not done:
            if np.random.rand() < self.epsilon:
                legal = self.env.get_legal_actions()
                if not legal: break
                action = random.choice(legal)
            else:
                action, _ = self.get_best_action(self.env)

            if action is None: break

            self.env.perform_action(action)
            next_state = self.get_state(self.env)

            curr_score = get_total_score_on_board(self.env.board_matrix, self.env.cat_tiles)

            reward = (curr_score - prev_score) * 10.0
            prev_score = curr_score

            done = self.env.is_game_over()

            self.memory.append((curr_state, action, reward, next_state, done))

            loss_sum += self.replay()

            curr_state = next_state
            steps += 1

        if self.epsilon > self.min_epsilon:
            self.epsilon *= self.decay

        return prev_score, loss_sum / max(1, steps), init_pred_real


if __name__ == "__main__":
    env = CalicoEnv()
    agent = FullTDAgent(env)
    viz = TrainingVisualizer()

    if agent.vn.load("full_calico.pkl"):
        print("Loaded weights.")
        agent.epsilon = 0.5

    print("Starting Full Calico Training (10k Episodes) with Normalization...")
    episodes = 1000
    scores = []
    score = 1

    try:
        for e in range(episodes):
            s, l, p = agent.train_episode()
            scores.append(s)

            if e % 10 == 0:
                avg = np.mean(scores[-50:])
                filters = agent.vn.conv.filters
                print(f"Ep {e} | Score: {s} | Avg: {avg:.2f} | Eps: {agent.epsilon:.2f} | Pred: {p:.1f}")
                viz.update(e, s, avg, l, p, s, filters)

            if e % 100 == 0:
                agent.vn.save("full_calico.pkl")

    except KeyboardInterrupt:
        print("\nTraining interrupted by user. Saving progress...")
    finally:
        agent.vn.save("full_calico.pkl")
        print("Model saved.")
        if not HEADLESS:
            plt.ioff()
            plt.show()