import copy
import random
import numpy as np

# --- MOCK CONSTANTS (In case you run this standalone) ---
# If you have your own utils, these will just be overwritten or used as fallbacks.
try:
    from enviroment.calico_scoring import get_total_score_on_board
    from utils.mini_calico_constants import *
except ImportError:
    # Defaults for demonstration if files are missing
    print("Warning: Importing local files failed. Using default constants.")
    BOARD_SIZE = 5
    PLAYER_HAND_SIZE = 2
    NR_OF_TILES_IN_SHOP = 3
    NO_TILE_VALUE = -1
    OBJECTIVE_VALUE_BASE = -10
    TILE_COLORS = 3
    TILE_PATTERNS = 3
    NR_OF_IDENTICAL_TILES = 3
    OBJECTIVE_POSITIONS_ON_BOARD = [(2, 2)]


    def get_total_score_on_board(matrix, cats):
        return 0

BOARD_BORDERS = {
    "mini": [0, 4, 6, 1, 5, 7, 2, 3, 8, 0, 7, 2, 3, 6, 5, 7]
}


# --- VISUALIZATION CONSTANTS ---
class Colors:
    RESET = "\033[0m"
    BOLD = "\033[1m"
    GRAY = "\033[90m"

    # Foreground colors for tiles
    RED = "\033[91m"
    GREEN = "\033[92m"
    YELLOW = "\033[93m"
    BLUE = "\033[94m"
    MAGENTA = "\033[95m"
    CYAN = "\033[96m"

    PALETTE = [RED, BLUE, YELLOW, GREEN, MAGENTA, CYAN]


# Symbols to represent the patterns (0, 1, 2...)
PATTERNS = ["●", "✚", "▲", "■", "♦", "*"]


class MiniCalicoEnv:
    def __init__(self):
        self.tile_pool = None
        self.player_tiles = []
        self.shop_tiles = []
        self.cat_tiles = []
        self.board_matrix = []
        self.move_history = []
        self.mode = ""
        self.selected_player_tile_index = 0

    def start_game(self, seed=None):
        if seed:
            random.seed(seed)
            np.random.seed(seed)

        self.tile_pool = self.initiate_tile_pool()
        self.player_tiles = [self.generate_random_tile() for _ in range(PLAYER_HAND_SIZE)]
        self.shop_tiles = [self.generate_random_tile() for _ in range(NR_OF_TILES_IN_SHOP)]
        self.cat_tiles = np.array([1, 2, 3])
        self.board_matrix = self.initialize_inner_board()
        self.initialize_outer_board()
        self.mode = "placing"
        self.selected_player_tile_index = 0

    def get_legal_actions(self):
        legal_actions = []
        if self.mode == "placing":
            valid_hand_indices = [i for i, x in enumerate(self.player_tiles) if x != NO_TILE_VALUE]
            rows, cols = np.where(self.board_matrix[1:BOARD_SIZE - 1, 1:BOARD_SIZE - 1] == NO_TILE_VALUE)
            rows += 1
            cols += 1
            for idx in valid_hand_indices:
                for r, c in zip(rows, cols):
                    legal_actions.append(('place', idx, r, c))
        elif self.mode == "buying":
            for shop_idx, _ in enumerate(self.shop_tiles):
                legal_actions.append(('buy', shop_idx, None, None))
        return legal_actions

    def fill_board_randomly(self):
        for i in range(BOARD_SIZE-2):
            for j in range(BOARD_SIZE-2):
                if [i+1,j+1] in OBJECTIVE_POSITIONS_ON_BOARD:
                    continue
                self.board_matrix[i+1][j+1] = self.generate_random_tile()

    def perform_action(self, action):
        action_type, tile_idx, row, col = action
        if action_type == 'place':
            self.place_tile(row, col, tile_idx)
        elif action_type == 'buy':
            self.buy_tile(tile_idx)

    def place_tile(self, row, col, hand_idx):
        tile_id = self.player_tiles[hand_idx]
        self.board_matrix[row][col] = tile_id
        self.player_tiles[hand_idx] = NO_TILE_VALUE
        self.selected_player_tile_index = hand_idx
        self.mode = "buying"

    def buy_tile(self, shop_idx):
        new_tile = self.shop_tiles[shop_idx]
        self.player_tiles[self.selected_player_tile_index] = new_tile
        draw_tile = self.generate_random_tile()
        if draw_tile is None: draw_tile = -99
        self.shop_tiles[shop_idx] = draw_tile
        self.mode = "placing"

    def initiate_tile_pool(self):
        pool = np.full(TILE_COLORS * TILE_PATTERNS, NR_OF_IDENTICAL_TILES)
        return pool

    def generate_random_tile(self):
        valid = np.where(self.tile_pool > 0)[0]
        if len(valid) == 0: return None
        choice = int(np.random.choice(valid))
        self.tile_pool[choice] -= 1
        return choice

    def initialize_inner_board(self):
        matrix = np.full((BOARD_SIZE, BOARD_SIZE), NO_TILE_VALUE, dtype=int)
        for i, pos in enumerate(OBJECTIVE_POSITIONS_ON_BOARD):
            r, c = pos
            matrix[r][c] = OBJECTIVE_VALUE_BASE - i
        return matrix

    def initialize_outer_board(self):
        border = BOARD_BORDERS["mini"]
        b_idx = 0
        # Top
        for c in range(BOARD_SIZE):
            self.board_matrix[0][c] = border[b_idx % len(border)]
            b_idx += 1
        # Right
        for r in range(1, BOARD_SIZE):
            self.board_matrix[r][BOARD_SIZE - 1] = border[b_idx % len(border)]
            b_idx += 1
        # Bottom
        for c in range(BOARD_SIZE - 2, -1, -1):
            self.board_matrix[BOARD_SIZE - 1][c] = border[b_idx % len(border)]
            b_idx += 1
        # Left
        for r in range(BOARD_SIZE - 2, 0, -1):
            self.board_matrix[r][0] = border[b_idx % len(border)]
            b_idx += 1

    def calculate_score(self):
        return get_total_score_on_board(self.board_matrix, self.cat_tiles)

    def _get_tile_str(self, val):
        """Helper to format a single tile value into a colored string."""
        if val == NO_TILE_VALUE:
            return f"{Colors.GRAY} . {Colors.RESET}"

        if val <= OBJECTIVE_VALUE_BASE:
            # Objective (e.g., -10 becomes "Obj")
            return f"{Colors.BOLD}OBJ{Colors.RESET}"

        if val == -99:
            return "XXX"

        # Standard Tile
        color_idx = val // TILE_PATTERNS
        pattern_idx = val % TILE_PATTERNS

        # Safety check for index out of bounds
        c_code = Colors.PALETTE[color_idx % len(Colors.PALETTE)]
        sym = PATTERNS[pattern_idx % len(PATTERNS)]

        return f"{c_code} {sym} {Colors.RESET}"

    def render(self):
        """Prints the board, hand, and shop in a nice format."""

        print(f"\n{Colors.BOLD}=== MINI CALICO BOARD ==={Colors.RESET}")

        # 1. Print The Board Matrix
        rows, cols = self.board_matrix.shape

        for r in range(rows):
            # --- THE HEX SHIFT LOGIC ---
            # If row is odd (1, 3, 5), add 2 spaces indentation.
            # 2 spaces roughly aligns the bracket `[` between the two brackets above it.
            indent = "   " if r % 2 == 1 else ""

            line_str = f"{r:2} {indent}"  # Row Number + Indent

            for c in range(cols):
                val = self.board_matrix[r][c]
                # Added a space " " after the bracket block to let the grid breathe
                line_str += f"[{self._get_tile_str(val)}] "

            print(line_str)
        # 2. Print Game State Info
        print(f"\n{Colors.BOLD}State:{Colors.RESET} {self.mode.upper()}")

        # 3. Print Hand
        hand_str = " ".join([f"[{self._get_tile_str(t)}]" for t in self.player_tiles])
        print(f"{Colors.BOLD}Player Hand:{Colors.RESET} {hand_str}")

        # 4. Print Shop
        shop_str = " ".join([f"[{self._get_tile_str(t)}]" for t in self.shop_tiles])
        print(f"{Colors.BOLD}Market:{Colors.RESET}      {shop_str}")
        print(f"{Colors.BOLD}Score:{Colors.RESET}      {str(self.calculate_score())}")
        print("=========================\n")


    def __str__(self):
        # Override str so print(env) works automatically
        self.render()
        return ""


# --- DEMO EXECUTION ---
if __name__ == "__main__":
    env = MiniCalicoEnv()
    env.start_game()

    print("1. Initial State:")
    print(env)

    # Find a valid move to demonstrate change
    legal = env.get_legal_actions()
    if legal:
        action = legal[0]
        print(f"Performing Action: {action}")
        env.perform_action(action)
        print(env)

    # Buy something
    if env.mode == "buying":
        shop_action = ("buy", 0, None, None)
        print(f"Performing Buy: {shop_action}")
        env.perform_action(shop_action)
        print(env)

    env.fill_board_randomly()
    print("2. Final State:")
    print(env)
