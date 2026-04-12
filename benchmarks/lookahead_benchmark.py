import copy
import random
import os
import csv
import time
import numpy as np

# --- PROJECT IMPORTS ---
# Adjust these paths if your project structure is slightly different
try:
    from enviroment.calico_env import CalicoEnv
    from enviroment.calico_scoring import get_total_score_on_board, get_total_score_on_board_detailed
    # We assume evaluate_move is your heuristic function (e.g. potential score)
    # If you don't have this file yet, you can replace it with get_total_score_on_board
    from enviroment.calico_potential_scoring import evaluate_move, generate_random_config
    from utils.constants import *
    from visualize.plots import plot_score_distribution, plot_average_convergence
except ImportError as e:
    print(f"Import Error: {e}")
    print("Please ensure you are running this from the root directory.")
    exit()

# --- CONFIGURATION ---
LOG_RESULTS_TO_CSV = True
LOG_FILENAME = "datasets/agent_lookahead_results_test_cat.csv"
EVALUATION_CONFIG = {}  # Dictionary for your potential scoring weights


# --- LOGGING HELPER ---
def log_game_result(score_details: dict):
    """Logs the detailed score of a single game run to a CSV file."""
    if not LOG_RESULTS_TO_CSV:
        return

    # Ensure directory exists
    os.makedirs(os.path.dirname(LOG_FILENAME), exist_ok=True)

    file_exists = os.path.exists(LOG_FILENAME)
    fieldnames = ['timestamp', 'score', 'objectives_score', 'cats_score', 'color_score']

    try:
        with open(LOG_FILENAME, 'a', newline='') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            if not file_exists or os.path.getsize(LOG_FILENAME) == 0:
                writer.writeheader()

            writer.writerow({
                'timestamp': score_details['timestamp'],
                'score': score_details['score'],
                'objectives_score': score_details['objectives_score'],
                'cats_score': score_details['cats_score'],
                'color_score': score_details['color_score']
            })
    except IOError as e:
        print(f"Error logging to CSV: {e}")


# --- AGENT LOGIC ---

def get_best_move_recursive(env, depth):
    """
    Finds the best move by looking ahead 'depth' steps.
    Returns: (best_score, best_action)
    """
    legal_actions = env.get_legal_actions()

    # Base Case: No more depth or Game Over
    if depth == 0 or not legal_actions:
        # Evaluate the board state "as is"
        # Note: evaluate_move should return a heuristic score (float/int)
        current_score = evaluate_move(env.board_matrix, env.cat_tiles)
        return current_score, None

    best_score = -float('inf')
    best_action = None

    # Recursive Step
    for action in legal_actions:
        # 1. Do Move
        env.perform_action(action)

        # 2. Recurse (Get the score of the best future path)
        score, _ = get_best_move_recursive(env, depth - 1)

        # 3. Undo Move (Backtrack)
        env.undo_action()

        # 4. Compare
        if score > best_score:
            best_score = score
            best_action = action

    # Fallback if no actions improved score (rare)
    if best_action is None and legal_actions:
        best_action = random.choice(legal_actions)

    return best_score, best_action


def one_step_lookahead_agent(env, depth=1):
    """Wrapper to call the recursive search and perform the action."""
    _, best_action = get_best_move_recursive(env, depth)

    if best_action:
        env.perform_action(best_action)
    else:
        # Fallback if no moves possible (e.g. end of game logic glitch)
        pass


# --- GAME LOOP ---

def run_lookahead_game(depth=1, print_board=False):
    env = CalicoEnv()
    env.start_game()

    while not env.is_game_over():
        one_step_lookahead_agent(env, depth=depth)

    if print_board:
        print(f"Game Over. Final Board:\n{env}")

    return env


# --- BENCHMARK RUNNER ---

def test_lookahead_benchmark(number_of_tries=100, depth=1, print_results=True):
    print(f"\n--- Starting Benchmark: Depth {depth} Lookahead ({number_of_tries} games) ---")

    total_metrics = {'score': 0, 'cats': 0, 'color': 0, 'objs': 0, 'time': 0}
    all_scores = []
    avg_convergence = []
    max_score = -1
    best_board_str = ""

    for i in range(number_of_tries):
        start_time = time.time()

        # Run Game
        env = run_lookahead_game(depth=depth, print_board=(i % 50 == 0))

        duration = time.time() - start_time

        # Calculate Scores
        total, cats, color, objs = get_total_score_on_board_detailed(env.board_matrix, env.cat_tiles)

        # Logging
        log_data = {
            'timestamp': time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()),
            'score': total,
            'objectives_score': objs,
            'cats_score': cats,
            'color_score': color
        }
        log_game_result(log_data)

        # Update Stats
        total_metrics['score'] += total
        total_metrics['cats'] += cats
        total_metrics['color'] += color
        total_metrics['objs'] += objs
        total_metrics['time'] += duration

        all_scores.append(total)
        avg_convergence.append(total_metrics['score'] / (i + 1))

        if total > max_score:
            max_score = total
            best_board_str = str(env)

        if (i + 1) % 10 == 0:
            print(f"Game {i + 1}/{number_of_tries} | Score: {total} | Avg so far: {avg_convergence[-1]:.2f}")

    # Final Statistics
    avg_metrics = {k: v / number_of_tries for k, v in total_metrics.items()}

    if print_results:
        print("\n" + "=" * 30)
        print("BENCHMARK RESULTS")
        print("=" * 30)
        print(f"Max Score Achieved: {max_score}")
        print(f"Best Board:\n{best_board_str}")
        print("-" * 20)
        print(f"Avg Total Score:      {avg_metrics['score']:.2f}")
        print(f"Avg Objective Score:  {avg_metrics['objs']:.2f}")
        print(f"Avg Cat Score:        {avg_metrics['cats']:.2f}")
        print(f"Avg Color Score:      {avg_metrics['color']:.2f}")
        print(f"Avg Time per Game:    {avg_metrics['time']:.4f} sec")
        print(f"Total Benchmark Time: {total_metrics['time']:.2f} sec")

        # Visualization
        try:
            plot_score_distribution(all_scores)
            plot_average_convergence(avg_convergence)
            print("Plots generated successfully.")
        except Exception as e:
            print(f"Plotting failed: {e}")

    return avg_metrics['score']


if __name__ == "__main__":
    # Run 50 games with depth 1 (One Step Lookahead)
    test_lookahead_benchmark(number_of_tries=50, depth=1)

    # Uncomment to try Depth 2 (Warning: Will be much slower!)
    # test_lookahead_benchmark(number_of_tries=10, depth=2)