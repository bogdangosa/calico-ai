import random

PLACING_MODE = 1
BUYING_MODE = 0

class RandomAgent:
    def __init__(self):
        pass

    def place_random_tile(self, state):
        placed_tile = random.randint(0, 1)

        return placed_tile

    def buy_random_tile(self, state):
        bought_tile = random.randint(0, 1)
        return Action

    def select_action(self, state):
        mode = state[0]
        if mode == PLACING_MODE:
            return self.place_random_tile(state)
        else:
            return self.buy_random_tile(state)