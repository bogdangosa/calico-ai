import copy
import random

import numpy as np

from enviroment.calico_env import CalicoEnv
from enviroment.calico_potential_scoring import evaluate_move, generate_random_config
from enviroment.calico_scoring import get_total_score_on_board, get_total_score_on_board_detailed
from utils.constants import *
from visualize.plots import plot_score_distribution, plot_average_convergence


def get_random_tile_to_buy():
    return random.randint(0,NR_OF_TILES_IN_SHOP-1)

def buy_random_tile(env):
    bought_tile_index = get_random_tile_to_buy()
    bought_tile_id = env.buy_tile(bought_tile_index)
    return bought_tile_id

def one_step_lookahead_move(env, remaining_recursions=0):
    max_score = -1
    best_action = None

    for action in env.get_legal_actions():

        env.perform_action(action)
        has_legal_actions_remaining = len(env.get_legal_actions()) > 0

        if remaining_recursions > 0 and has_legal_actions_remaining:
            one_step_lookahead_move(env, remaining_recursions - 1)

        score = evaluate_move(env.board_matrix, env.cat_tiles)

        if remaining_recursions > 0 and has_legal_actions_remaining:
            env.undo_action()

        env.undo_action()

        if score > max_score:
            max_score = score
            best_action = action
    if best_action:
        env.perform_action(best_action)
    return best_action

def one_step_lookahead_move_old(env,remaining_recursions=0):
    if env.mode == "buying":
        tile_to_buy = -1
        max_score = -1
        for tile_index in range(NR_OF_TILES_IN_SHOP):
            temp_env = copy.deepcopy(env)
            temp_env.buy_tile(tile_index)
            one_step_lookahead_move(temp_env)
            score = evaluate_move(temp_env.board_matrix, temp_env.cat_tiles)
            if score > max_score:
                max_score = score
                tile_to_buy = tile_index
        env.buy_tile(tile_to_buy)
        return tile_to_buy
    elif env.mode == "placing":
        max_score = -1
        selected_move = (0,0,0)
        for cat_tile_index in range(PLAYER_HAND_SIZE):
            for row in range(1,BOARD_SIZE-1):
                for col in range(1,BOARD_SIZE-1):
                    tile_id = env.board_matrix[row][col]
                    if tile_id != NO_TILE_VALUE:
                        continue
                    temp_env = copy.deepcopy(env)
                    temp_env.place_tile(row,col,cat_tile_index)
                    score = evaluate_move(temp_env.board_matrix,temp_env.cat_tiles)
                    if score > max_score:
                        max_score = score
                        selected_move = row,col,cat_tile_index
        env.place_tile(selected_move[0],selected_move[1],selected_move[2])
        return selected_move
    return None


def run_one_step_lookahead_game(print_board=False):
    env = CalicoEnv()
    env.start_game()

    while not env.is_game_over():
        one_step_lookahead_move(env,1)
        #print("score "+str(get_total_score_on_board(env.board_matrix, env.cat_tiles)))
    if print_board:
        print(env)
        print("score " + str(get_total_score_on_board(env.board_matrix, env.cat_tiles)))
    return env

def test_one_step_lookahead(number_of_tries=100,config=EVALUATION_CONFIG,print_results=False):
    import time

    total_score = 0
    total_cats = 0
    total_color = 0
    total_objectives = 0
    total_time = 0
    best_table = None
    max_score = 0
    all_scores = []
    average_scores = []

    for i in range(number_of_tries):
        start = time.time()
        env = run_one_step_lookahead_game(i % 10 == 0)

        score, cats_score, color_score, objectives_score = get_total_score_on_board_detailed(env.board_matrix,
                                                                                             env.cat_tiles)

        # update best
        if score > max_score:
            max_score = score
            best_table = env

        # accumulate totals
        total_score += score
        total_cats += cats_score
        total_color += color_score
        total_objectives += objectives_score
        total_time += time.time() - start

        if print_results:
            all_scores.append(score)
            average_scores.append(total_score /(i+1))

    # print results
    if print_results:
        print("Best board:\n", best_table)
        print("Max score:", max_score)
        print("Average total score:", total_score / number_of_tries)
        print("Average cats score:", total_cats / number_of_tries)
        print("Average color score:", total_color / number_of_tries)
        print("Average objectives score:", total_objectives / number_of_tries)
        print("Average time per game:", total_time / number_of_tries, "seconds")
        print("Total time:", total_time, "seconds")
        plot_score_distribution(all_scores)  # bar chart for distribution
        plot_average_convergence(average_scores)  # line chart for convergence
    return total_score / number_of_tries


def test_random_configs(
    num_configs=10,
    number_of_tries=100,
    min_val=0.0,
    max_val=5.0
):
    results = []

    for i in range(num_configs):
        config = generate_random_config(min_val, max_val)
        avg_score = test_one_step_lookahead(number_of_tries, config)
        results.append((avg_score, config))

        print(f"[{i+1}/{num_configs}] avg_score={avg_score:.2f}  config={config}")

    # Sort best → worst
    results.sort(key=lambda x: x[0], reverse=True)

    print("\n===== BEST CONFIGS =====")
    for score, cfg in results[:5]:
        print(f"Score={score:.2f}  Config={cfg}")

    return results

#test_random_configs()
if __name__ == "__main__":
    print(test_one_step_lookahead(number_of_tries=100,print_results=True))
