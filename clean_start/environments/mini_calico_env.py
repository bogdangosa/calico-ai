from typing import List

import numpy as np

from clean_start.environments.scheme import CalicoAction
from enviroment.calico_scoring import flood_fill
from enviroment.mini_calico_env import Colors

PLAYER_HAND_SIZE=2
TILE_PATTERNS = 3
TILE_COLORS=3
NR_OF_IDENTICAL_TILES=2
BOARD_SIZE = 2
NO_TILE_VALUE = 37
CAT_TILE_TYPES = 2

PATTERNS = ["●", "✚", "▲", "■", "♦", "*"]

def get_color_id(tile_id):
    return tile_id // TILE_COLORS

def get_pattern_id(tile_id):
    return tile_id % TILE_PATTERNS

class MicroCalicoEnv:
    def __init__(self):
        self.tile_pool = None
        self.player_tiles = []
        self.cat_tiles = []
        self.board_matrix = []
        self.move_history = []
        self.selected_player_tile_index = 0

    def initialize_cat_tiles(self):
        cat_tiles = np.arange(1, CAT_TILE_TYPES + 1)
        np.random.shuffle(cat_tiles)
        return cat_tiles

    def start_game(self, seed=41):
        self.tile_pool = self.initiate_tile_pool()
        self.player_tiles = [self.generate_random_tile() for _ in range(PLAYER_HAND_SIZE)]
        self.cat_tiles = self.initialize_cat_tiles()
        self.board_matrix = self.initialize_board()
        self.board_matrix = self.initialize_board()
        self.selected_player_tile_index = 0

    def get_legal_actions(self) -> List[CalicoAction]:
        """Return a list of legal actions as tuples:
           (action_type, tile_index, row, col)
           action_type: 'place' or 'buy'
           For 'buy', row and col are None
        """
        legal_actions = []
        for tile_idx in range(len(self.player_tiles)):
            for row in range(BOARD_SIZE):
                for col in range(BOARD_SIZE):
                    if self.board_matrix[row][col] == NO_TILE_VALUE:
                        legal_actions.append(CalicoAction(
                            action_type="place",
                            tile_index=tile_idx,
                            row=row,
                            col=col
                        ))
        return legal_actions

    def perform_place_action(self, action: CalicoAction):
        selected_tile = self.player_tiles[action.tile_index]
        self.board_matrix[action.row][action.col] = selected_tile
        self.player_tiles = []
        self.player_tiles = [self.generate_random_tile() for _ in range(PLAYER_HAND_SIZE)]

    def perform_buy_action(self, action: CalicoAction):
        pass

    def perform_action(self, action: CalicoAction):
        if action.action_type=="place":
            self.perform_place_action(action)
        else:
            self.perform_buy_action(action)


    def generate_random_tile(self):
        """Randomly pick a tile ID from pool."""
        valid_indices = np.where(self.tile_pool > 0)[0]
        if len(valid_indices) == 0:
            return None
        tile_id = int(np.random.choice(valid_indices))
        self.tile_pool[tile_id] -= 1
        return tile_id

    def initiate_tile_pool(self):
        pool = np.full(TILE_COLORS * TILE_PATTERNS, NR_OF_IDENTICAL_TILES)
        return pool

    def initialize_board(self):
        matrix = np.full((BOARD_SIZE, BOARD_SIZE), NO_TILE_VALUE, dtype=int)
        return matrix

    def fill_board_randomly(self):
        for i in range(BOARD_SIZE):
            for j in range(BOARD_SIZE):
                self.board_matrix[i][j] = self.generate_random_tile()

    def _get_tile_str(self, val):
        """Helper to format a single tile value into a colored string."""
        if val == NO_TILE_VALUE:
            return f"{Colors.GRAY} . {Colors.RESET}"

        if val == -99:
            return "XXX"

        # Standard Tile
        color_idx = val // TILE_PATTERNS
        pattern_idx = val % TILE_PATTERNS

        # Safety check for index out of bounds
        c_code = Colors.PALETTE[color_idx % len(Colors.PALETTE)]
        sym = PATTERNS[pattern_idx % len(PATTERNS)]

        return f"{c_code} {sym} {Colors.RESET}"

    def evaluate_board(self):
        """
            Evaluates the board based on Micro Calico rules:
            - 1 point for each color group >= 2 tiles.
            - 2 points for each pattern group >= 3 tiles.
            """
        color_score = 0
        cat_score = 0

        visited_colors = {}
        visited_patterns = {}

        for y in range(BOARD_SIZE):
            for x in range(BOARD_SIZE):
                tile_val = self.board_matrix[y][x]

                if tile_val == NO_TILE_VALUE or tile_val < 0:

                    continue

                if f"{y},{x}" not in visited_colors:
                    color_region = flood_fill(self.board_matrix, y, x, visited_colors, get_color_id)
                    if len(color_region) >= 2:
                        color_score += 1

                if f"{y},{x}" not in visited_patterns:
                    pattern_id = get_pattern_id(tile_val)
                    pattern_region = flood_fill(self.board_matrix, y, x, visited_patterns, get_pattern_id)
                    if len(pattern_region) >= 3 and (pattern_id + 1) in self.cat_tiles:
                        cat_score += 2

        total_score = color_score + cat_score
        return total_score, cat_score, color_score


    def render(self):
        """Prints the board, hand, and shop in a nice format."""

        print(f"\n{Colors.BOLD}=== MINI CALICO BOARD ==={Colors.RESET}")
        rows, cols = self.board_matrix.shape

        for r in range(rows):
            indent = "   " if r % 2 == 1 else ""

            line_str = f"{r:2} {indent}"  # Row Number + Indent

            for c in range(cols):
                val = self.board_matrix[r][c]
                line_str += f"[{self._get_tile_str(val)}] "

            print(line_str)

        cat_symbols = [PATTERNS[p - 1] for p in self.cat_tiles]
        print(f"{Colors.BOLD}Cats like patterns:{Colors.RESET} {' '.join(cat_symbols)}")

        hand_str = " ".join([f"[{self._get_tile_str(t)}]" for t in self.player_tiles])
        print(f"{Colors.BOLD}Player Hand:{Colors.RESET} {hand_str}")
        print(f"{Colors.BOLD}Current score:{self.evaluate_board()[0]}")

        print("=========================\n")

env = MicroCalicoEnv()


env.start_game()


env.render()

env.fill_board_randomly()


env.render()