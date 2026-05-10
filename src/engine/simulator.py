import time
from datetime import datetime

import numpy as np
import pandas as pd
from loguru import logger

from src.engine.scoring.scoring import ScoringCalculator
from src.utils.datasets import generate_short_id, save_simulation_details, append_to_results
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
            raise RuntimeError("No action selected.")

        env.perform_action(action)

    return scorer.get_total_detailed_score(
        env.board_matrix,
        env.cat_tiles
    )


def run_simulation(env, agent, num_games=1000, progress_interval=100,plot_score_interval=True,save_to_dataset=False):
    """Runs a batch of games and reports metrics."""
    game_data = []
    logger.info(f"Starting simulation: {num_games} games with {agent.__class__.__name__}...")

    start_time = time.perf_counter()

    for i in range(num_games):
        total, color, obj, cat = play_full_game(env, agent)
        game_data.append({
            "total_score": total,
            "color_score": color,
            "objective_score": obj,
            "cat_score": cat
        })

        if (i + 1) % progress_interval == 0:
            logger.info(f"Completed {i + 1}/{num_games} games...")

    duration = time.perf_counter() - start_time
    total_scores = [g["total_score"] for g in game_data]

    average_score = np.mean(total_scores)
    max_score = np.max(total_scores)
    avg_ms_per_game = (duration / num_games) * 1000

    logger.info("\n" + "=" * 30)
    logger.info("SIMULATION COMPLETE")
    logger.info(f"Average Score: {average_score:.2f}")
    logger.info(f"Highest Score: {max_score}")
    logger.info(f"Total Time:    {duration:.4f}s")
    logger.info(f"Avg Time/Game: {avg_ms_per_game:.2f}ms")
    logger.info("=" * 30 + "\n")
    if plot_score_interval:
        plot_score_distribution(total_scores)
    if save_to_dataset:
        agent_name = agent.__class__.__name__
        sim_id = generate_short_id()
        env_name = env.config.name

        sim_path = f"../datasets/{env_name}/{agent_name}/simulation_{sim_id}.csv"
        save_simulation_details(game_data, sim_path)

        df_temp = pd.DataFrame(game_data)
        summary = {
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "sim_id": sim_id,
            "agent": agent_name,
            "avg_total": df_temp["total_score"].mean(),
            "avg_color": df_temp["color_score"].mean(),
            "avg_obj": df_temp["objective_score"].mean(),
            "avg_cat": df_temp["cat_score"].mean(),
            "avg_ms_per_game": avg_ms_per_game,
            "num_games": num_games,
            **agent.get_hyperparameters()
        }

        results_path = f"../datasets/{env_name}/results.csv"
        append_to_results(summary, results_path)
    return total_scores