import numpy as np
import pickle
import os
from enviroment.calico_env import CalicoEnv
from enviroment.calico_scoring import get_total_score_on_board_detailed
from experiments.deep_q_learning import FullCNNValueNetwork


def run_standalone_evaluation(model_path="hybrid_calico_model.pkl", num_games=100):
    env = CalicoEnv()
    vn = FullCNNValueNetwork(channels=13)  # Match training channels

    # Load the saved weights
    if not os.path.exists(model_path):
        print(f"Error: {model_path} not found.")
        return

    with open(model_path, 'rb') as f:
        w = pickle.load(f)
        vn.conv.filters, vn.conv.bias = w['c'], w['cb']
        vn.W1, vn.b1, vn.W2, vn.b2 = w['w1'], w['b1'], w['w2'], w['b2']

    print(f"Loaded weights. Starting {num_games} standalone games...")
    scores = []

    for g in range(num_games):
        env.start_game()
        while not env.is_game_over():
            legal_actions = env.get_legal_actions()
            best_val, best_action = -float('inf'), None

            for action in legal_actions:
                env.perform_action(action)
                # Forward pass: CNN evaluates the board's potential
                val = vn.forward(env.get_board_tensor())
                env.undo_action()

                if val > best_val:
                    best_val, best_action = val, action

            if best_action:
                env.perform_action(best_action)

        final_score = get_total_score_on_board_detailed(env.board_matrix, env.cat_tiles)[0]
        scores.append(final_score)

        if (g + 1) % 2 == 0:
            print(f"Game {g + 1}/{num_games} | Score: {final_score} | Avg so far: {np.mean(scores):.2f}")

    print(f"\nEvaluation Complete.")
    print(f"Final Average Score: {np.mean(scores):.2f}")
    print(f"Max Score: {np.max(scores)} | Min Score: {np.min(scores)}")


if __name__ == "__main__":
    run_standalone_evaluation()