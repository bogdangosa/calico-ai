import copy
import random

from enviroment.calico_env import CalicoEnv
from enviroment.calico_potential_scoring import evaluate_move
from enviroment.calico_scoring import get_total_score_on_board, get_total_score_on_board_detailed
from utils.constants import *

def get_random_tile_to_buy():
    return random.randint(0,NR_OF_TILES_IN_SHOP-1)

def buy_random_tile(env):
    bought_tile_index = get_random_tile_to_buy()
    bought_tile_id = env.buy_tile(bought_tile_index)
    return bought_tile_id

def one_step_lookahead_move(env):
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
        one_step_lookahead_move(env)
        #print("score "+str(get_total_score_on_board(env.board_matrix, env.cat_tiles)))
    if print_board:
        print(env)
        print("score " + str(get_total_score_on_board(env.board_matrix, env.cat_tiles)))
    return env


import time

NR_OF_TRIES = 100
total_score = 0
total_cats = 0
total_color = 0
total_objectives = 0
total_time = 0
best_table = None
max_score = 0

for i in range(NR_OF_TRIES):
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

# print results
print("Best board:\n", best_table)
print("Max score:", max_score)
print("Average total score:", total_score / NR_OF_TRIES)
print("Average cats score:", total_cats / NR_OF_TRIES)
print("Average color score:", total_color / NR_OF_TRIES)
print("Average objectives score:", total_objectives / NR_OF_TRIES)
print("Average time per game:", total_time / NR_OF_TRIES, "seconds")
print("Total time:", total_time, "seconds")
