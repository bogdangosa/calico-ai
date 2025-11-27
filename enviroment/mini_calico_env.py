import copy
import random
import numpy as np

# --- MINI CALICO CONSTANTS ---
# Reduced complexity for debugging and initial learning
BOARD_SIZE = 5  # 5x5 Matrix (1-tile border, 3x3 playable area)
TILE_COLORS = 3  # Reduced from 6
TILE_PATTERNS = 3  # Reduced from 6
TOTAL_TILE_TYPES = TILE_COLORS * TILE_PATTERNS  # 9 unique tiles
NR_OF_IDENTICAL_TILES = 5  # 9 * 5 = 45 tiles total
NR_OF_TILES_IN_SHOP = 2
PLAYER_HAND_SIZE = 2
NO_TILE_VALUE = -1  # Using -1 for empty is safer for CNNs than 37
OBJECTIVE_VALUE_BASE = -100  # Objectives are negative

# Simple Objective in the center (Index 2,2)
OBJECTIVE_POSITIONS_ON_BOARD = [[2, 2]]

# Simplified Borders for 5x5 (Top, Right, Bottom, Left)
# Just random valid tile IDs for the border to allow matching
BOARD_BORDERS = {
    "mini": [0, 1, 2, 3, 4, 5, 6, 7, 8, 0, 1, 2, 3, 4, 5, 6]  # Simplified loop
}


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

        # Cats are simplified: just IDs for now
        self.cat_tiles = np.array([1, 2])

        self.board_matrix = self.initialize_inner_board()
        self.initialize_outer_board()
        self.mode = "placing"
        self.selected_player_tile_index = 0

    def get_legal_actions(self):
        legal_actions = []

        if self.mode == "placing":
            # Optimization: Only try to place valid tiles (not empty slots in hand)
            valid_hand_indices = [i for i, x in enumerate(self.player_tiles) if x != NO_TILE_VALUE]

            # Get all empty coordinates on the board
            # Only check 1 to BOARD_SIZE-1 to avoid borders
            rows, cols = np.where(self.board_matrix[1:BOARD_SIZE - 1, 1:BOARD_SIZE - 1] == NO_TILE_VALUE)
            # Offset by 1 because we sliced the center
            rows += 1
            cols += 1

            for idx in valid_hand_indices:
                for r, c in zip(rows, cols):
                    legal_actions.append(('place', idx, r, c))

        elif self.mode == "buying":
            for shop_idx, _ in enumerate(self.shop_tiles):
                legal_actions.append(('buy', shop_idx, None, None))

        return legal_actions

    def perform_action(self, action):
        action_type, tile_idx, row, col = action

        # Record state for undo
        record = {
            "action_type": action_type,
            "mode": self.mode,
            "selected_player_tile_index": self.selected_player_tile_index,
            "player_tiles": list(self.player_tiles),  # Copy list
            "shop_tiles": list(self.shop_tiles),  # Copy list
        }

        if action_type == 'place':
            record["prev_board_val"] = self.board_matrix[row][col]
            record["row"] = row
            record["col"] = col
            self.place_tile(row, col, tile_idx)

        elif action_type == 'buy':
            # For buying, we need to know exactly what tile was removed/added to undo perfectly
            # But for simple learning, full state restoration (above) is easier
            self.buy_tile(tile_idx)

        self.move_history.append(record)

    def undo_action(self):
        if not self.move_history:
            return

        record = self.move_history.pop()

        # Restore general state
        self.mode = record["mode"]
        self.selected_player_tile_index = record["selected_player_tile_index"]
        self.player_tiles = record["player_tiles"]
        self.shop_tiles = record["shop_tiles"]

        # Specific restore for place
        if record["action_type"] == 'place':
            r, c = record["row"], record["col"]
            self.board_matrix[r][c] = record["prev_board_val"]

    def place_tile(self, row, col, hand_idx):
        tile_id = self.player_tiles[hand_idx]
        self.board_matrix[row][col] = tile_id
        self.player_tiles[hand_idx] = NO_TILE_VALUE
        self.selected_player_tile_index = hand_idx
        self.mode = "buying"

    def buy_tile(self, shop_idx):
        # Take tile from shop to hand
        new_tile = self.shop_tiles[shop_idx]
        self.player_tiles[self.selected_player_tile_index] = new_tile

        # Refill shop
        draw_tile = self.generate_random_tile()
        if draw_tile is None: draw_tile = -99  # Empty pool marker

        # Replace the specific shop slot
        self.shop_tiles[shop_idx] = draw_tile
        self.mode = "placing"

    # --- INITIALIZATION HELPERS ---

    def initiate_tile_pool(self):
        # 3 Colors * 3 Patterns * N identical copies
        pool = np.full(TILE_COLORS * TILE_PATTERNS, NR_OF_IDENTICAL_TILES)
        return pool

    def generate_random_tile(self):
        valid = np.where(self.tile_pool > 0)[0]
        if len(valid) == 0: return None
        choice = int(np.random.choice(valid))
        self.tile_pool[choice] -= 1
        return choice

    def initialize_inner_board(self):
        # Initialize with NO_TILE_VALUE
        matrix = np.full((BOARD_SIZE, BOARD_SIZE), NO_TILE_VALUE, dtype=int)
        # Place Objectives
        for i, pos in enumerate(OBJECTIVE_POSITIONS_ON_BOARD):
            r, c = pos
            matrix[r][c] = OBJECTIVE_VALUE_BASE - i
        return matrix

    def initialize_outer_board(self):
        # Fill borders with simplified pattern
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

    def is_game_over(self):
        # Check playable area (1 to 3)
        inner = self.board_matrix[1:BOARD_SIZE - 1, 1:BOARD_SIZE - 1]
        # Game over if no empty spots (-1) remain
        return not np.any(inner == NO_TILE_VALUE)

    # --- SIMPLIFIED SCORING FOR LEARNING ---
    # We calculate score locally to avoid dependency on the big complex scoring file
    # which assumes a 7x7 board.

    def calculate_score(self):
        """
        Simplified scoring for Mini Calico:
        +1 point for every neighbor sharing COLOR
        +1 point for every neighbor sharing PATTERN
        """
        score = 0

        # Iterate over inner playable board
        for r in range(1, BOARD_SIZE - 1):
            for c in range(1, BOARD_SIZE - 1):
                tile_id = self.board_matrix[r][c]

                # Skip empty or objective
                if tile_id < 0: continue

                my_color = tile_id // TILE_PATTERNS
                my_pattern = tile_id % TILE_PATTERNS

                # Check 4 neighbors (Up, Down, Left, Right)
                neighbors = [(r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)]

                for nr, nc in neighbors:
                    n_id = self.board_matrix[nr][nc]
                    if n_id < 0: continue  # Skip empty/obj neighbors

                    n_color = n_id // TILE_PATTERNS
                    n_pattern = n_id % TILE_PATTERNS

                    if n_color == my_color: score += 1
                    if n_pattern == my_pattern: score += 1

        return score

    # --- STATE FOR CNN ---

    def get_cnn_state(self):
        """
        Returns (H, W, 4) for CNN input.
        Channels: [Color, Pattern, IsOccupied, IsObjective]
        """
        H, W = self.board_matrix.shape
        state = np.zeros((H, W, 4), dtype=float)

        for r in range(H):
            for c in range(W):
                val = self.board_matrix[r][c]

                if val >= 0:
                    # Tile
                    color = val // TILE_PATTERNS
                    pattern = val % TILE_PATTERNS

                    state[r, c, 0] = (color + 1) / float(TILE_COLORS)
                    state[r, c, 1] = (pattern + 1) / float(TILE_PATTERNS)
                    state[r, c, 2] = 1.0  # Occupied
                elif val <= OBJECTIVE_VALUE_BASE:
                    # Objective
                    state[r, c, 3] = 1.0
                # Else: Empty (-1) -> all zeros

        return state

    def get_flat_state(self):
        # Fallback if using dense network
        return self.get_cnn_state().flatten()

    def __str__(self):
        return str(self.board_matrix)


if __name__ == "__main__":
    env = MiniCalicoEnv()
    env.start_game()
    print("Mini Calico Initialized")
    env.perform_action(("place",1,1,1))
    env.perform_action(("buy",1,1,1))
    env.perform_action(("place",1,2,1))
    print(env)
    print("Score:", env.calculate_score())