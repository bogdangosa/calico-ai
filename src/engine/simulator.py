import time
import numpy as np
from loguru import logger

from src.engine.scoring.scoring import ScoringCalculator
from src.utils.timing import time_it
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
            logger.info(env)

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
    logger.info(f"Starting simulation: {num_games} games with {agent.__class__.__name__}...")

    start_time = time.perf_counter()

    for i in range(num_games):
        score = play_full_game(env, agent)
        scores.append(score)

        if (i + 1) % progress_interval == 0:
            logger.info(f"Completed {i + 1}/{num_games} games...")

    duration = time.perf_counter() - start_time

    average_score = np.mean(scores)
    max_score = np.max(scores)

    logger.info("\n" + "=" * 30)
    logger.info("SIMULATION COMPLETE")
    logger.info(f"Average Score: {average_score:.2f}")
    logger.info(f"Highest Score: {max_score}")
    logger.info(f"Total Time:    {duration:.4f}s")
    logger.info(f"Avg Time/Game: {(duration / num_games) * 1000:.2f}ms")
    logger.info("=" * 30 + "\n")

    plot_score_distribution(scores)
    return scores