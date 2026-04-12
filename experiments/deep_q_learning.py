import numpy as np
import random
import pickle
import os
import copy
import matplotlib
from collections import deque

# --- GUI / Headless Setup ---
try:
    import tkinter

    matplotlib.use('TkAgg')
    HEADLESS = False
except ImportError:
    matplotlib.use('Agg')
    HEADLESS = True
import matplotlib.pyplot as plt

# --- Environment & Constants ---
from enviroment.calico_scoring import get_total_score_on_board
from utils.constants import *

SCORE_MAX = 120.0
GAMMA = 0.95
BATCH_SIZE = 64
TARGET_UPDATE_FREQ = 10


# --- Visualization Helper ---
class TrainingVisualizer:
    def __init__(self):
        self.plot_ready = False
        self.episodes, self.scores, self.losses = [], [], []
        self.preds, self.acts = [], []

        try:
            if not HEADLESS: plt.ion()
            self.fig, self.axs = plt.subplots(2, 2, figsize=(12, 8))
            if not HEADLESS:
                self.fig.canvas.manager.set_window_title('DQN Calico Training Dashboard')
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
            self.text_obj = self.ax_stats.text(0.05, 0.5, "Initializing...", fontsize=11, family='monospace')

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

            # Update Trends
            self.line_score.set_data(self.episodes, self.scores)
            self.line_loss.set_data(self.episodes, self.losses)
            self.ax_trends.relim();
            self.ax_trends.autoscale_view()
            self.ax_trends_loss.relim();
            self.ax_trends_loss.autoscale_view()
            self.ax_trends.set_title(f"DQN Learning (Ep {episode})")

            # Update Estimation Accuracy
            self.scat.set_offsets(np.c_[self.preds, self.acts])
            if self.preds:
                mn, mx = min(min(self.preds), min(self.acts)), max(max(self.preds), max(self.acts))
                if mn == mx: mx += 1
                self.line_ideal.set_data([mn, mx], [mn, mx])
                self.ax_acc.set_xlim(mn, mx);
                self.ax_acc.set_ylim(mn, mx)
            self.ax_acc.set_title("Value Prediction Accuracy")

            # Update Filters
            if filters is not None:
                n = min(4, filters.shape[0])
                imgs = [(filters[i, :, :, 0] - filters[i, :, :, 0].min()) / (
                            filters[i, :, :, 0].max() - filters[i, :, :, 0].min() + 1e-5) for i in range(n)]
                combined = np.hstack(imgs)
                if self.img_obj is None:
                    self.img_obj = self.ax_filt.imshow(combined, cmap='viridis')
                    self.ax_filt.axis('off')
                else:
                    self.img_obj.set_data(combined)
            self.ax_filt.set_title("Conv Layer Filters (Channel 0)")

            stats = (f"Ep: {episode}\nScore: {score}\nAvg(50): {avg_score:.2f}\nLoss: {loss:.4f}\n"
                     f"Pred: {pred:.1f} | Act: {act:.1f}")
            self.text_obj.set_text(stats)

            if HEADLESS:
                plt.savefig("dqn_calico_dashboard.png")
            else:
                self.fig.canvas.flush_events()
        except Exception as e:
            print(f"Viz Update Error: {e}")


# --- Calico Environment ---
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

    def start_game(self, seed=None):
        if seed: random.seed(seed); np.random.seed(seed)
        self.tile_pool = np.full(TILE_COLORS * TILE_PATTERNS, NR_OF_IDENTICAL_TILES)
        self.player_tiles = [self.generate_random_tile() for _ in range(PLAYER_HAND_SIZE)]
        self.shop_tiles = [self.generate_random_tile() for _ in range(NR_OF_TILES_IN_SHOP)]
        self.cat_tiles = np.arange(1, CAT_TILE_TYPES + 1)
        np.random.shuffle(self.cat_tiles)
        self.board_matrix = np.full((BOARD_SIZE, BOARD_SIZE), NO_TILE_VALUE, dtype=int)
        for i, (row, col) in enumerate(OBJECTIVE_POSITIONS_ON_BOARD, start=1):
            self.board_matrix[row, col] = -i
        self.initialize_outer_board("purple")
        self.mode = "placing"
        self.selected_player_tile_index = 0

    def generate_random_tile(self):
        valid_indices = np.where(self.tile_pool > 0)[0]
        if len(valid_indices) == 0: return 0
        tile_id = int(np.random.choice(valid_indices));
        self.tile_pool[tile_id] -= 1
        return tile_id

    def initialize_outer_board(self, board_color):
        board_border = BOARD_BORDERS[board_color];
        idx = 0
        for c in range(BOARD_SIZE): self.board_matrix[0][c] = board_border[idx]; idx += 1
        for r in range(1, BOARD_SIZE - 1): self.board_matrix[r][BOARD_SIZE - 1] = board_border[idx]; idx += 1
        for c in range(BOARD_SIZE - 1, -1, -1): self.board_matrix[BOARD_SIZE - 1][c] = board_border[idx]; idx += 1
        for r in range(BOARD_SIZE - 2, 0, -1): self.board_matrix[r][0] = board_border[idx]; idx += 1

    def get_legal_actions(self):
        legal_actions = []
        if self.mode == "placing":
            rows, cols = np.where(self.board_matrix != NO_TILE_VALUE)
            potential_spots = set()
            for r, c in zip(rows, cols):
                for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                    nr, nc = r + dr, c + dc
                    if 1 <= nr < BOARD_SIZE - 1 and 1 <= nc < BOARD_SIZE - 1:
                        if self.board_matrix[nr][nc] == NO_TILE_VALUE: potential_spots.add((nr, nc))
            empty_cells = list(potential_spots) if potential_spots else [(3, 3)]
            for idx, tile_id in enumerate(self.player_tiles):
                if tile_id == NO_TILE_VALUE: continue
                for r, c in empty_cells: legal_actions.append(('place', idx, r, c))
        elif self.mode == "buying":
            for shop_idx in range(len(self.shop_tiles)): legal_actions.append(('buy', shop_idx, None, None))
        return legal_actions

    def perform_action(self, action):
        a_type, tile_idx, r, c = action
        if a_type == 'place':
            self.move_history.append(("place", r, c, tile_idx, self.player_tiles[tile_idx], self.mode))
            self.board_matrix[r][c] = self.player_tiles[tile_idx]
            self.player_tiles[tile_idx] = NO_TILE_VALUE
            self.selected_player_tile_index = tile_idx;
            self.mode = "buying"
        else:
            self.move_history.append(
                ("buy", self.shop_tiles.copy(), self.player_tiles[self.selected_player_tile_index], self.mode))
            self.player_tiles[self.selected_player_tile_index] = self.shop_tiles[tile_idx]
            self.shop_tiles[tile_idx] = self.generate_random_tile();
            self.mode = "placing"

    def undo_action(self):
        record = self.move_history.pop()
        if record[0] == "place":
            _, r, c, idx, val, mode = record
            self.board_matrix[r][c] = NO_TILE_VALUE;
            self.player_tiles[idx] = val;
            self.mode = mode
        else:
            _, shop_snap, hand_val, mode = record
            self.shop_tiles = shop_snap;
            self.player_tiles[self.selected_player_tile_index] = hand_val;
            self.mode = mode

    def get_board_tensor(self):
        tensor = np.zeros((BOARD_SIZE, BOARD_SIZE, TILE_COLORS + TILE_PATTERNS + 1))
        for r in range(BOARD_SIZE):
            for c in range(BOARD_SIZE):
                val = self.board_matrix[r, c]
                if val >= 0:
                    tensor[r, c, val // TILE_PATTERNS] = 1.0
                    tensor[r, c, TILE_COLORS + (val % TILE_PATTERNS)] = 1.0
                elif val < 0 and val != NO_TILE_VALUE:
                    tensor[r, c, 12] = 1.0
        return tensor


# --- CNN Layers ---
class Conv2DLayer:
    def __init__(self, num_filters, f_size, in_ch, lr):
        self.filters = np.random.randn(num_filters, f_size, f_size, in_ch) * np.sqrt(2 / (f_size * f_size * in_ch))
        self.bias = np.zeros(num_filters);
        self.lr = lr

    def forward(self, X):
        self.last_input = X
        h, w, _ = X.shape;
        out = np.zeros((h - 2, w - 2, self.filters.shape[0]))
        for i in range(h - 2):
            for j in range(w - 2):
                region = X[i:i + 3, j:j + 3]
                for f in range(self.filters.shape[0]):
                    out[i, j, f] = np.sum(region * self.filters[f]) + self.bias[f]
        return out

    def backward(self, d_out):
        d_f = np.zeros_like(self.filters)
        for i in range(d_out.shape[0]):
            for j in range(d_out.shape[1]):
                region = self.last_input[i:i + 3, j:j + 3]
                for f in range(self.filters.shape[0]): d_f[f] += d_out[i, j, f] * region
        self.filters -= self.lr * np.clip(d_f, -0.1, 0.1);
        self.bias -= self.lr * np.sum(d_out, axis=(0, 1))


class FullCNNValueNetwork:
    def __init__(self, channels, lr=0.02):
        self.conv = Conv2DLayer(32, 3, channels, lr)
        self.W1 = np.random.randn(5 * 5 * 32, 128) * np.sqrt(2 / (5 * 5 * 32))
        self.b1 = np.zeros(128);
        self.W2 = np.random.randn(128, 1) * np.sqrt(2 / 128);
        self.b2 = np.array([0.25]);
        self.lr = lr

    def forward(self, X):
        self.c_out = self.conv.forward(X);
        self.flat = self.c_out.flatten()
        self.z1 = np.dot(self.flat, self.W1) + self.b1;
        self.a1 = np.maximum(0, self.z1)
        return (np.dot(self.a1, self.W2) + self.b2)[0]

    def backward(self, X, target):
        pred = self.forward(X);
        grad = np.clip(2 * (pred - target), -1, 1)
        dW2 = np.outer(self.a1, grad);
        db2 = grad;
        da1 = grad * self.W2.flatten()
        dz1 = da1 * (self.z1 > 0);
        dW1 = np.outer(self.flat, dz1);
        db1 = dz1
        self.W1 -= self.lr * np.clip(dW1, -1, 1);
        self.b1 -= self.lr * db1
        self.W2 -= self.lr * dW2;
        self.b2 -= self.lr * db2
        self.conv.backward(np.dot(dz1, self.W1.T).reshape(5, 5, 32))
        return (pred - target) ** 2


# --- DQN Agent ---
class DQNAgent:
    def __init__(self, env, decay=0.9967):
        self.env = env
        self.policy_net = FullCNNValueNetwork(13)
        self.target_net = FullCNNValueNetwork(13)
        self.update_target()
        self.memory = deque(maxlen=5000);
        self.epsilon = 1.0;
        self.decay = decay;
        self.steps = 0

    def update_target(self):
        self.target_net.conv.filters = np.copy(self.policy_net.conv.filters)
        self.target_net.W1, self.target_net.b1 = np.copy(self.policy_net.W1), np.copy(self.policy_net.b1)
        self.target_net.W2, self.target_net.b2 = np.copy(self.policy_net.W2), np.copy(self.policy_net.b2)

    def select_action(self):
        legal = self.env.get_legal_actions()
        if random.random() < self.epsilon: return random.choice(legal)
        best_q, best_a = -float('inf'), None
        for a in legal:
            self.env.perform_action(a);
            q = self.policy_net.forward(self.env.get_board_tensor());
            self.env.undo_action()
            if q > best_q: best_q, best_a = q, a
        return best_a

    def train_episode(self):
        self.env.start_game()
        curr_state = self.env.get_board_tensor()
        init_pred_norm = self.policy_net.forward(curr_state)
        init_pred_real = init_pred_norm * SCORE_MAX

        prev_score, loss_sum, count = 0, 0, 0
        while not np.all(self.env.board_matrix != NO_TILE_VALUE):
            s = self.env.get_board_tensor();
            a = self.select_action()
            if a is None: break
            self.env.perform_action(a);
            s_p = self.env.get_board_tensor()
            curr_score = get_total_score_on_board(self.env.board_matrix, self.env.cat_tiles)
            r = (curr_score - prev_score) * 10.0;
            prev_score = curr_score
            done = np.all(self.env.board_matrix != NO_TILE_VALUE)
            self.memory.append((s, r / SCORE_MAX, s_p, done))
            if len(self.memory) > BATCH_SIZE:
                batch = random.sample(self.memory, BATCH_SIZE)
                for ms, mr, msp, md in batch:
                    target = mr if md else mr + GAMMA * self.target_net.forward(msp)
                    loss_sum += self.policy_net.backward(ms, target)
                count += 1
            if done: break
        self.steps += 1
        if self.steps % TARGET_UPDATE_FREQ == 0: self.update_target()
        self.epsilon = max(0.1, self.epsilon * self.decay)
        return curr_score, loss_sum / max(1, count), init_pred_real


if __name__ == "__main__":
    env = CalicoEnv();
    agent = DQNAgent(env, decay=0.9967);
    viz = TrainingVisualizer()
    scores = [];
    print("Training DQN Calico with Dashboard...")
    for e in range(1001):
        s, l, p = agent.train_episode();
        scores.append(s)
        if e % 10 == 0:
            avg = np.mean(scores[-50:])
            print(f"Ep {e} | Score: {s} | Avg50: {avg:.2f} | Eps: {agent.epsilon:.2f}")
            viz.update(e, s, avg, l, p, s, agent.policy_net.conv.filters)
    if not HEADLESS: plt.ioff(); plt.show()