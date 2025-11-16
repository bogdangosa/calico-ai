import random

from enviroment.calico_env import CalicoEnv
from enviroment.calico_scoring import get_total_score_on_board
from utils.constants import *

PLACING_MODE = 1
BUYING_MODE = 0

class RandomAgent:
    def __init__(self):
        pass

    def place_random_tile(self, state):
        placed_tile = random.randint(0, 1)

        return placed_tile

    def buy_random_tile(self, state):
        pass

    def select_action(self, state):
        mode = state[0]
        if mode == PLACING_MODE:
            return self.place_random_tile(state)
        else:
            return self.buy_random_tile(state)

def get_random_tile_from_hand():
    return random.randint(0,PLAYER_HAND_SIZE-1)

def get_random_tile_to_buy():
    return random.randint(0,NR_OF_TILES_IN_SHOP-1)


NR_OF_TRIES = 100
average = 0
for i in range(NR_OF_TRIES):
    env = CalicoEnv()
    env.start_game()
    env.fill_board_randomly()
    score = get_total_score_on_board(env.board_matrix, env.cat_tiles)
    average += score
    if i % 10 == 0:
        print(f"Played {i} games")
        print(env)
average /= NR_OF_TRIES
print(average)
