import time
import numpy as np
from src.engine.scoring import ScoringCalculator
from src.utils.visuals import plot_score_distribution

def play_full_game(env, agent, render=False):
    """
    A universal runner for any Calico agent.

    Args:
        env: The Calico environment instance.
        agent: Any object with a .select_action(env) method.
        render: If True, prints the board state at every turn.
    """
    env.start_game()
    scorer = ScoringCalculator(env.config)

    while not env.is_game_over():
        if render:
            print(env)

        action = agent.select_action(env)

        if action is None:
            break

        env.perform_action(action)

    final_score, *details = scorer.get_total_detailed_score(
        env.board_matrix,
        env.cat_tiles
    )

    return final_score


def run_simulation(env, agent, num_games=1000, progress_interval=100):
    """Runs a batch of games and reports metrics."""
    scores = []
    print(f"Starting simulation: {num_games} games with {agent.__class__.__name__}...")

    start_time = time.perf_counter()

    for i in range(num_games):
        score = play_full_game(env, agent)
        scores.append(score)

        if (i + 1) % progress_interval == 0:
            print(f"Completed {i + 1}/{num_games} games...")

    duration = time.perf_counter() - start_time

    average_score = np.mean(scores)
    max_score = np.max(scores)

    print("\n" + "=" * 30)
    print("SIMULATION COMPLETE")
    print(f"Average Score: {average_score:.2f}")
    print(f"Highest Score: {max_score}")
    print(f"Total Time:    {duration:.4f}s")
    print(f"Avg Time/Game: {(duration / num_games) * 1000:.2f}ms")
    print("=" * 30 + "\n")

    plot_score_distribution(scores)
    return scores