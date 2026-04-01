import numpy as np
import random
import pickle
import os
import copy
import matplotlib
from collections import deque

try:
    import tkinter

    matplotlib.use('TkAgg')
    HEADLESS = False
except ImportError:
    matplotlib.use('Agg')
    HEADLESS = True
import matplotlib.pyplot as plt

from enviroment.calico_env import CalicoEnv
from enviroment.calico_potential_scoring import evaluate_move
from enviroment.calico_scoring import get_total_score_on_board
from external_solutions.deep_q_learning import FullCNNValueNetwork

SCORE_MAX = 120.0
GAMMA = 0.98
BATCH_SIZE = 64

class TrainingVisualizer:
    def __init__(self):
        self.plot_ready = False
        self.episodes, self.scores, self.losses = [], [], []
        self.preds, self.acts = [], []
        try:
            if not HEADLESS: plt.ion()
            self.fig, self.axs = plt.subplots(2, 2, figsize=(12, 8))
            if not HEADLESS:
                self.fig.canvas.manager.set_window_title('Hybrid Hindsight Training Dashboard')
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
            self.line_score.set_data(self.episodes, self.scores)
            self.line_loss.set_data(self.episodes, self.losses)
            self.ax_trends.relim();
            self.ax_trends.autoscale_view()
            self.ax_trends_loss.relim();
            self.ax_trends_loss.autoscale_view()
            self.scat.set_offsets(np.c_[self.preds, self.acts])
            if self.preds:
                mn, mx = min(min(self.preds), min(self.acts)), max(max(self.preds), max(self.acts))
                if mn == mx: mx += 1
                self.line_ideal.set_data([mn, mx], [mn, mx])
                self.ax_acc.set_xlim(mn, mx);
                self.ax_acc.set_ylim(mn, mx)
            if filters is not None:
                n = min(4, filters.shape[0])
                imgs = [(filters[i, :, :, 0] - filters[i, :, :, 0].min()) / (
                            filters[i, :, :, 0].max() - filters[i, :, :, 0].min() + 1e-5) for i in range(n)]
                combined = np.hstack(imgs)
                if self.img_obj is None:
                    self.img_obj = self.ax_filt.imshow(combined, cmap='viridis'); self.ax_filt.axis('off')
                else:
                    self.img_obj.set_data(combined)
            stats = (f"Ep: {episode}\nScore: {score}\nAvg(50): {avg_score:.2f}\nLoss: {loss:.6f}\n"
                     f"Pred (Expert): {pred:.1f} | Act: {act:.1f}")
            self.text_obj.set_text(stats)
            if HEADLESS:
                plt.savefig("hybrid_hindsight_dashboard.png")
            else:
                self.fig.canvas.flush_events()
        except Exception as e:
            print(f"Viz Update Error: {e}")


class HybridHindsightAgent:
    def __init__(self, env, decay=0.9967):
        self.env = env
        self.vn = FullCNNValueNetwork(channels=13, lr=0.01)
        self.memory = deque(maxlen=10000)
        self.epsilon = 1.0
        self.decay = decay
        self.min_epsilon = 0.1

    def save_model(self, filename="hybrid_calico_model.pkl"):
        """ Extracts weights and saves to a pickle file. """
        weights = {
            'c': self.vn.conv.filters,
            'cb': self.vn.conv.bias,
            'w1': self.vn.W1,
            'b1': self.vn.b1,
            'w2': self.vn.W2,
            'b2': self.vn.b2
        }
        with open(filename, 'wb') as f:
            pickle.dump(weights, f)
        print(f"--- Model weights successfully saved to {filename} ---")

    def get_expert_advice(self, env):
        max_score = -1
        best_action = None
        legal_actions = env.get_legal_actions()
        if not legal_actions: return None, 0
        for action in legal_actions:
            env.perform_action(action)
            score = evaluate_move(env.board_matrix, env.cat_tiles)
            env.undo_action()
            if score > max_score:
                max_score = score
                best_action = action
        return best_action, max_score

    def train_step(self):
        if len(self.memory) < BATCH_SIZE: return 0
        batch = random.sample(self.memory, BATCH_SIZE)
        loss_sum = 0
        for state, action, expert_potential, next_state, done in batch:
            target = expert_potential / SCORE_MAX
            loss_sum += self.vn.backward(state, target)
        return loss_sum / BATCH_SIZE

    def train_episode(self):
        self.env.start_game()
        total_loss, steps = 0, 0
        init_state = self.env.get_board_tensor()
        init_pred = self.vn.forward(init_state) * SCORE_MAX
        while not self.env.is_game_over():
            state = self.env.get_board_tensor()
            best_action, expert_potential = self.get_expert_advice(self.env)
            if random.random() < self.epsilon:
                legal = self.env.get_legal_actions()
                action = random.choice(legal) if legal else None
            else:
                action = best_action
            if action is None: break
            self.env.perform_action(action)
            next_state = self.env.get_board_tensor()
            self.memory.append((state, action, expert_potential, next_state, self.env.is_game_over()))
            total_loss += self.train_step()
            steps += 1
        self.epsilon = max(self.min_epsilon, self.epsilon * self.decay)
        final_score = get_total_score_on_board(self.env.board_matrix, self.env.cat_tiles)
        return final_score, total_loss / max(1, steps), init_pred


if __name__ == "__main__":
    env = CalicoEnv()
    agent = HybridHindsightAgent(env, decay=0.997)
    viz = TrainingVisualizer()
    scores = []

    print("Starting Hybrid Hindsight Training with Dashboard...")
    try:
        for e in range(1, 1001):
            score, loss, pred = agent.train_episode()
            scores.append(score)
            if e % 5 == 0:
                avg = np.mean(scores[-50:])
                print(f"Episode {e} | Score: {score} | Avg: {avg:.2f} | Loss: {loss:.6f}")
                viz.update(e, score, avg, loss, pred, score, agent.vn.conv.filters)
    except KeyboardInterrupt:
        print("\nTraining interrupted. Saving current progress...")
    finally:
        # --- SAVE THE MODEL HERE ---
        agent.save_model("hybrid_calico_model.pkl")
        if not HEADLESS:
            plt.ioff()
            plt.show()