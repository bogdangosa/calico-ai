import numpy as np
import random
import pickle  # Used for saving/loading the "brain" (weights) of the AI
import os
import time

# --- ROBUST MATPLOTLIB SETUP ---
import matplotlib

try:
    # Try to import tkinter to see if the system supports the standard GUI
    import tkinter

    matplotlib.use('TkAgg')
    HEADLESS = False
except ImportError:
    # If failed (e.g., missing libtk8.6.so), fallback to Agg (Headless)
    print("Warning: GUI libraries not found. Switching to Headless mode (Saving to PNG).")
    matplotlib.use('Agg')
    HEADLESS = True

import matplotlib.pyplot as plt
from collections import deque
from enviroment.mini_calico_env import MiniCalicoEnv, BOARD_SIZE, NO_TILE_VALUE, TILE_COLORS, TILE_PATTERNS


# ==============================================================================
# PART 0: VISUALIZATION HELPER (ROBUST & MODULAR)
# ==============================================================================
class TrainingVisualizer:
    """
    Helper class to draw graphs.
    Supports both Live Window (GUI) and File Saving (Headless).
    """

    def __init__(self):
        self.plot_ready = False
        # Data buffers
        self.episodes = []
        self.scores = []
        self.losses = []
        self.preds = []
        self.acts = []

        try:
            if not HEADLESS:
                plt.ion()  # Interactive mode only if GUI exists

            self.fig, self.axs = plt.subplots(2, 2, figsize=(12, 8))

            if not HEADLESS:
                self.fig.canvas.manager.set_window_title('Deep RL Training Dashboard')

            # Pre-calculate axes
            self.ax_trends_score = self.axs[0, 0]
            self.ax_trends_loss = self.ax_trends_score.twinx()
            self.ax_accuracy = self.axs[0, 1]
            self.ax_filters = self.axs[1, 0]
            self.ax_stats = self.axs[1, 1]

            if not HEADLESS:
                plt.tight_layout()
                plt.show(block=False)

            self.plot_ready = True
        except Exception as e:
            print(f"[Visualizer Init Error] {e}")

    def update(self, episode, score, avg_score, loss, predicted_val, actual_val, filters):
        if not self.plot_ready: return

        # 1. Update Data safely
        try:
            self.episodes.append(episode)
            self.scores.append(avg_score)
            self.losses.append(loss)
            self.preds.append(float(predicted_val))
            self.acts.append(float(actual_val))

            if len(self.episodes) > 200:
                self.episodes.pop(0)
                self.scores.pop(0)
                self.losses.pop(0)
                self.preds.pop(0)
                self.acts.pop(0)
        except Exception as e:
            print(f"[Visualizer Data Error] {e}")
            return

        # 2. Plotting
        self.plot_trends(episode)
        self.plot_prediction_accuracy()
        self.plot_filters(filters)
        self.plot_stats(episode, score, avg_score, loss, predicted_val, actual_val)

        # 3. Render or Save
        try:
            plt.tight_layout()

            if HEADLESS:
                # Save to disk instead of showing window
                plt.savefig("training_dashboard.png")
            else:
                # Live update
                plt.draw()
                plt.pause(0.1)
        except Exception as e:
            print(f"[Visualizer Render Error] {e}")

    def plot_trends(self, episode):
        try:
            self.ax_trends_score.clear()
            self.ax_trends_loss.clear()

            self.ax_trends_score.plot(self.episodes, self.scores, label='Avg Score', color='blue', linewidth=2)
            self.ax_trends_score.set_ylabel('Score', color='blue', fontsize=10)
            self.ax_trends_score.set_xlabel('Episode')
            self.ax_trends_score.set_title(f"Learning Curve (Ep {episode})", fontsize=10)
            self.ax_trends_score.grid(True, alpha=0.3)

            self.ax_trends_loss.plot(self.episodes, self.losses, label='Loss', color='red', alpha=0.3, linewidth=1)
            self.ax_trends_loss.set_ylabel('Loss', color='red', fontsize=10)
        except Exception as e:
            print(f"[Visualizer Error] Plot Trends: {e}")

    def plot_prediction_accuracy(self):
        try:
            self.ax_accuracy.clear()
            self.ax_accuracy.scatter(self.preds, self.acts, alpha=0.6, c='purple', edgecolors='w')

            if self.preds and self.acts:
                low = min(min(self.preds), min(self.acts))
                high = max(max(self.preds), max(self.acts))
                if low == high:
                    low -= 1
                    high += 1
                self.ax_accuracy.plot([low, high], [low, high], 'k--', alpha=0.5, label='Ideal')

            self.ax_accuracy.set_xlabel("AI Predicted Score")
            self.ax_accuracy.set_ylabel("Actual Score")
            self.ax_accuracy.set_title("Value Estimation Accuracy")
            self.ax_accuracy.grid(True, alpha=0.3)
        except Exception as e:
            print(f"[Visualizer Error] Plot Accuracy: {e}")

    def plot_filters(self, filters):
        try:
            self.ax_filters.clear()
            self.ax_filters.axis('off')
            self.ax_filters.set_title("CNN Filters (Color Channel)")

            if filters is None or filters.size == 0: return

            n_filters = min(4, filters.shape[0])
            filter_imgs = []

            f_min, f_max = filters.min(), filters.max()

            for i in range(n_filters):
                w = filters[i, :, :, 0]
                if f_max - f_min > 1e-5:
                    w = (w - f_min) / (f_max - f_min)
                filter_imgs.append(w)

            if filter_imgs:
                sep = np.ones((3, 1)) * 0.5
                combined = filter_imgs[0]
                for img in filter_imgs[1:]:
                    combined = np.hstack((combined, sep, img))

                self.ax_filters.imshow(combined, cmap='viridis', interpolation='nearest')
        except Exception as e:
            print(f"[Visualizer Error] Plot Filters: {e}")

    def plot_stats(self, episode, score, avg_score, loss, predicted_val, actual_val):
        try:
            self.ax_stats.clear()
            self.ax_stats.axis('off')

            stat_text = (
                f"Episode: {episode}\n"
                f"Last Score: {score}\n"
                f"Avg Score (50): {avg_score:.2f}\n"
                f"Avg Loss: {loss:.4f}\n"
                f"Pred vs Actual:\n {predicted_val:.1f} / {actual_val:.1f}"
            )

            self.ax_stats.text(0.1, 0.4, stat_text, fontsize=12, fontfamily='monospace')
        except Exception as e:
            print(f"[Visualizer Error] Plot Stats: {e}")


# ==============================================================================
# PART 1: CONVOLUTIONAL NEURAL NETWORK (Built from Scratch & Optimized)
# ==============================================================================

class Conv2DLayer:
    def __init__(self, num_filters, filter_size, input_channels, learning_rate=0.01):
        self.num_filters = num_filters
        self.filter_size = filter_size
        self.input_channels = input_channels
        self.lr = learning_rate

        scale = np.sqrt(2.0 / (filter_size * filter_size * input_channels))
        self.filters = np.random.randn(num_filters, filter_size, filter_size, input_channels) * scale
        self.bias = np.zeros(num_filters)

    def iterate_regions(self, image):
        h, w, _ = image.shape
        for i in range(h - self.filter_size + 1):
            for j in range(w - self.filter_size + 1):
                im_region = image[i:(i + self.filter_size), j:(j + self.filter_size)]
                yield i, j, im_region

    def forward(self, input_data):
        self.last_input = input_data
        h, w, _ = input_data.shape
        output_h = h - self.filter_size + 1
        output_w = w - self.filter_size + 1
        output = np.zeros((output_h, output_w, self.num_filters))

        for i, j, im_region in self.iterate_regions(input_data):
            # Vectorized dot product over all filters at once
            for f in range(self.num_filters):
                output[i, j, f] = np.sum(im_region * self.filters[f]) + self.bias[f]
        return output

    def backward(self, d_L_d_out):
        d_L_d_filters = np.zeros(self.filters.shape)

        for i in range(d_L_d_out.shape[0]):
            for j in range(d_L_d_out.shape[1]):
                im_region = self.last_input[i:i + self.filter_size, j:j + self.filter_size]
                for f in range(self.num_filters):
                    d_L_d_filters[f] += d_L_d_out[i, j, f] * im_region

        # Gradient Clipping
        np.clip(d_L_d_filters, -0.1, 0.1, out=d_L_d_filters)

        self.filters -= self.lr * d_L_d_filters
        self.bias -= self.lr * np.sum(d_L_d_out, axis=(0, 1))
        return None


class MiniCNNValueNetwork:
    def __init__(self, board_size, channels, learning_rate=0.01):
        self.conv = Conv2DLayer(num_filters=16, filter_size=3, input_channels=channels, learning_rate=learning_rate)
        conv_out_dim = board_size - 2
        self.flat_size = conv_out_dim * conv_out_dim * 16
        self.W1 = np.random.randn(self.flat_size, 64) * np.sqrt(2 / self.flat_size)
        self.b1 = np.zeros(64)
        self.W2 = np.random.randn(64, 1) * np.sqrt(2 / 64)
        self.b2 = np.zeros(1)
        self.lr = learning_rate

    def forward(self, X):
        out_conv = self.conv.forward(X)
        self.out_conv_shape = out_conv.shape
        self.flat_input = out_conv.flatten()
        self.z1 = np.dot(self.flat_input, self.W1) + self.b1
        self.a1 = np.maximum(0, self.z1)
        self.z2 = np.dot(self.a1, self.W2) + self.b2
        self.output = self.z2[0]
        return self.output

    def backward(self, X, target):
        diff = self.output - target
        output_grad = 2 * diff
        output_grad = np.clip(output_grad, -1, 1)

        dW2 = np.outer(self.a1, output_grad)
        db2 = output_grad
        da1 = output_grad * self.W2.flatten()
        dz1 = da1 * (self.z1 > 0)
        dW1 = np.outer(self.flat_input, dz1)
        db1 = dz1
        d_flat = np.dot(dz1, self.W1.T)

        np.clip(dW1, -1, 1, out=dW1)
        np.clip(dW2, -1, 1, out=dW2)

        self.W1 -= self.lr * dW1
        self.b1 -= self.lr * db1
        self.W2 -= self.lr * dW2
        self.b2 -= self.lr * db2

        d_conv = d_flat.reshape(self.out_conv_shape)
        self.conv.backward(d_conv)

        return diff ** 2

    def save(self, filename="mini_cnn.pkl"):
        with open(filename, 'wb') as f:
            pickle.dump({
                'conv': self.conv.filters,
                'W1': self.W1, 'W2': self.W2,
                'b1': self.b1, 'b2': self.b2,
                'cb': self.conv.bias,
                'channels': self.conv.input_channels
            }, f)

    def load(self, filename="mini_cnn.pkl"):
        if os.path.exists(filename):
            try:
                with open(filename, 'rb') as f:
                    data = pickle.load(f)
                    saved_channels = data.get('channels', -1)
                    current_channels = self.conv.input_channels
                    if saved_channels != current_channels:
                        print(f"WARNING: Saved model has {saved_channels} channels, current has {current_channels}.")
                        print("Discarding incompatible save file.")
                        return False
                    self.conv.filters = data['conv']
                    self.conv.bias = data['cb']
                    self.W1 = data['W1']
                    self.W2 = data['W2']
                    self.b1 = data['b1']
                    self.b2 = data['b2']
                return True
            except Exception as e:
                print(f"Error loading file: {e}. Starting fresh.")
                return False
        return False


# ==============================================================================
# PART 2: TEMPORAL DIFFERENCE (TD) AGENT WITH EXPERIENCE REPLAY
# ==============================================================================

class MiniTDAgent:
    def __init__(self, env, gamma=0.90, epsilon=1.0, epsilon_min=0.1, epsilon_decay=0.997):
        self.env = env
        self.channels = TILE_COLORS + TILE_PATTERNS + 2
        self.vn = MiniCNNValueNetwork(BOARD_SIZE, self.channels)
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.memory = deque(maxlen=2000)
        self.batch_size = 32

    def get_better_cnn_state(self, env_instance):
        H, W = env_instance.board_matrix.shape
        state = np.zeros((H, W, self.channels), dtype=float)
        for r in range(H):
            for c in range(W):
                val = env_instance.board_matrix[r][c]
                if val >= 0:
                    color = val // TILE_PATTERNS
                    pattern = val % TILE_PATTERNS
                    state[r, c, color] = 1.0
                    state[r, c, TILE_COLORS + pattern] = 1.0
                    state[r, c, -2] = 1.0
                elif val <= -100:
                    state[r, c, -1] = 1.0
        return state

    def get_best_action(self, env):
        legal = env.get_legal_actions()
        if not legal: return None
        best_action = None
        best_value = -float('inf')
        for action in legal:
            env.perform_action(action)
            state = self.get_better_cnn_state(env)
            val = self.vn.forward(state)
            env.undo_action()
            if val > best_value:
                best_value = val
                best_action = action
        return best_action, best_value

    def replay(self):
        if len(self.memory) < self.batch_size:
            return 0
        batch = random.sample(self.memory, self.batch_size)
        batch_loss = 0
        for state, action, reward, next_state, done in batch:
            if done:
                target = reward
            else:
                target_val = self.vn.forward(next_state)
                target = reward + self.gamma * target_val
            loss = self.vn.backward(state, target)
            batch_loss += loss
        return batch_loss / len(batch)

    def train_episode(self):
        self.env.start_game()
        current_state = self.get_better_cnn_state(self.env)
        prev_score = self.env.calculate_score()
        initial_prediction = self.vn.forward(current_state)

        done = False
        total_loss = 0
        steps = 0

        while not done:
            if np.random.rand() < self.epsilon:
                legal = self.env.get_legal_actions()
                if not legal: break
                action = random.choice(legal)
            else:
                action, val = self.get_best_action(self.env)
            if action is None: break

            self.env.perform_action(action)
            next_state = self.get_better_cnn_state(self.env)

            current_score = self.env.calculate_score()
            reward = (current_score - prev_score) * 10
            prev_score = current_score
            done = self.env.is_game_over()

            self.memory.append((current_state, action, reward, next_state, done))
            loss = self.replay()
            total_loss += loss
            current_state = next_state
            steps += 1

        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

        return self.env.calculate_score(), total_loss / max(1, steps), initial_prediction


if __name__ == "__main__":
    env = MiniCalicoEnv()
    agent = MiniTDAgent(env)
    viz = TrainingVisualizer()

    if agent.vn.load("mini_cnn.pkl"):
        print("Loaded weights.")
        agent.epsilon = 0.5

    print("Starting Mini-Calico Training with Visuals...")
    if HEADLESS:
        print("(Running in HEADLESS mode. Check 'training_dashboard.png' for updates)")

    episodes = 3000
    scores = []

    for e in range(episodes):
        score, loss, initial_pred = agent.train_episode()
        scores.append(score)

        if e % 10 == 0:
            avg = np.mean(scores[-50:])
            filters = agent.vn.conv.filters
            print(f"Episode {e} | Score: {score} | Avg: {avg:.2f}")
            viz.update(e, score, avg, loss, initial_pred, score, filters)

        if e % 200 == 0:
            agent.vn.save("mini_cnn.pkl")

    agent.vn.save("mini_cnn.pkl")

    if not HEADLESS:
        plt.ioff()
        plt.show()