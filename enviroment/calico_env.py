import random

import numpy as np

from enviroment.calico_scoring import get_total_score_on_board
from utils.constants import *



class CalicoEnv:
    def __init__(self):
        self.tile_pool = None
        self.player_tiles = []
        self.shop_tiles = []
        self.cat_tiles = []
        self.board_matrix = []
        self.mode = ""
        self.selected_player_tile_index = 0

    #TEMP
    def get_legal_actions(self):
        """Return a list of legal actions as tuples:
           (action_type, tile_index, row, col)
           action_type: 'place' or 'buy'
           For 'buy', row and col are None
        """
        legal_actions = []
        if self.mode == "placing":
            # Place tile actions
            for idx, tile_id in enumerate(self.player_tiles):
                if tile_id == NO_TILE_VALUE:
                    continue
                for r in range(BOARD_SIZE):
                    for c in range(BOARD_SIZE):
                        if self.board_matrix[r][c] == NO_TILE_VALUE:
                            legal_actions.append(('place', idx, r, c))

        if self.mode == "buying":
            # Buy tile actions
            for shop_idx, shop_tile in enumerate(self.shop_tiles):
                legal_actions.append(('buy', shop_idx, None, None))

        return legal_actions

    def perform_action(self, action):
        """Apply an action to the environment."""
        action_type, tile_idx, row, col = action
        if action_type == 'place':
            self.place_tile(row, col, tile_idx)
        elif action_type == 'buy':
            self.buy_tile(tile_idx)

    #TEMP

    def start_game(self,seed=41):
        #random.seed(seed)
        #np.random.seed(seed)
        self.tile_pool = self.initiate_tile_pool()
        self.player_tiles = [self.generate_random_tile() for _ in range(PLAYER_HAND_SIZE)]
        self.shop_tiles = self.initiate_shop_tiles()
        self.cat_tiles = self.initialize_cat_tiles()
        self.board_matrix = self.initialize_inner_board()
        self.initialize_outer_board("purple")
        self.mode = "placing"
        self.selected_player_tile_index = NO_TILE_VALUE

    def is_game_over(self):
        return not np.any(self.board_matrix == NO_TILE_VALUE)

    def buy_tile(self, tile_index: int):
        """Buy a tile from the shop and update the player hand."""
        self.mode = "placing"
        bought_tile_id = self.shop_tiles[tile_index]
        self.player_tiles[self.selected_player_tile_index] = bought_tile_id
        self.replace_tile(tile_index)  # replace purchased tile
        self.replace_tile(0)  # optional: replace first tile in shop
        return bought_tile_id

    def place_tile(self, row: int, col: int, selected_tile_index: int):
        """Place a selected tile on the board and update score."""
        selected_tile_id = self.player_tiles[selected_tile_index]
        self.board_matrix[row][col] = selected_tile_id
        self.player_tiles[selected_tile_index] = NO_TILE_VALUE
        self.selected_player_tile_index = selected_tile_index
        self.mode = "buying"

    def replace_tile(self, tile_index: int):
        """
        Replace a tile in the shop at `tile_index` with a new random tile.
        """
        # Remove tile at tile_index
        self.shop_tiles = np.delete(self.shop_tiles, tile_index)

        # Generate a new random tile
        new_tile = self.generate_random_tile()

        # Append the new tile to the end of the shop
        self.shop_tiles = np.append(self.shop_tiles, new_tile)

    def initiate_tile_pool(self):
        return np.full(TILE_COLORS * TILE_PATTERNS, NR_OF_IDENTICAL_TILES)

    def initiate_shop_tiles(self):
        return [self.generate_random_tile() for _ in range(NR_OF_TILES_IN_SHOP)]

    def initiate_player_tiles(self):
        return [self.generate_random_tile() for _ in range(PLAYER_HAND_SIZE)]

    def initialize_cat_tiles(self):
        cat_tiles = np.arange(1, CAT_TILE_TYPES + 1)
        np.random.shuffle(cat_tiles)  # shuffles in place
        return cat_tiles

    def generate_random_tile(self):
        """Randomly pick a tile ID from pool."""
        valid_indices = np.where(self.tile_pool > 0)[0]
        if len(valid_indices) == 0:
            return None
        tile_id = int(np.random.choice(valid_indices))
        self.tile_pool[tile_id] -= 1
        return tile_id

    def fill_board_randomly(self):
        for i in range(BOARD_SIZE-2):
            for j in range(BOARD_SIZE-2):
                if [i,j] in OBJECTIVE_POSITIONS_ON_BOARD:
                    continue
                self.board_matrix[i+1][j+1] = self.generate_random_tile()

    def initialize_inner_board(self):
        """
        Creates a 5x5 playable board:
          - all cells initialized to NO_TILE_VALUE
          - objectives placed at specified coordinates (negative IDs)
        """
        board_matrix = np.full((BOARD_SIZE, BOARD_SIZE), NO_TILE_VALUE, dtype=int)

        # Place objectives (negative IDs)
        for i, (row, col) in enumerate(OBJECTIVE_POSITIONS_ON_BOARD, start=1):
            board_matrix[row, col] = -i

        return board_matrix

    def initialize_outer_board(self,board_color):
        board_border = BOARD_BORDERS[board_color]
        tile_index = 0

        # Top row (left to right): [0][0..6]
        for col in range(7):
            self.board_matrix[0][col] = board_border[tile_index]
            tile_index += 1

        # Right column (top to bottom, excluding corners): [1..5][6]
        for row in range(1, 6):
            self.board_matrix[row][6] = board_border[tile_index]
            tile_index += 1

        # Bottom row (right to left): [6][6..0]
        for col in range(6, -1, -1):
            self.board_matrix[6][col] = board_border[tile_index]
            tile_index += 1

        # Left column (bottom to top, excluding corners): [5..1][0]
        for row in range(5, 0, -1):
            self.board_matrix[row][0] = board_border[tile_index]
            tile_index += 1

    def get_flat_state(self):
        """
        Flattens all game components into a single 1D array:
          - tile_pool
          - player_tiles
          - shop_tiles
          - cat_tiles
          - board_matrix
        """
        state_parts = []
        # Append mode as integer (0 or 1)
        state_parts.append(np.array([int(self.mode == "placing")], dtype=int))
        # Player tiles
        state_parts.append(np.array(self.player_tiles, dtype=int))
        # Shop tiles
        state_parts.append(np.array(self.shop_tiles, dtype=int))
        # Cat tiles
        if isinstance(self.cat_tiles, np.ndarray):
            state_parts.append(self.cat_tiles.flatten())
        else:
            state_parts.append(np.array(self.cat_tiles, dtype=int))
        # Board matrix
        if isinstance(self.board_matrix, np.ndarray):
            state_parts.append(self.board_matrix.flatten())
        else:
            state_parts.append(np.array(self.board_matrix, dtype=int).flatten())
        # Concatenate everything into a single 1D array
        flat_state = np.concatenate(state_parts).astype(int)
        return flat_state

    def set_from_flat_state(self, flat_state):
        """
        Reconstructs the game state from a flattened 1D array produced by get_flat_state().
        """
        idx = 0

        # --- Mode ---
        self.mode = "placing" if flat_state[idx] == 1 else "other"
        idx += 1

        # --- Player tiles ---
        player_tile_count = len(self.player_tiles)
        self.player_tiles = flat_state[idx:idx + player_tile_count].tolist()
        idx += player_tile_count

        # --- Shop tiles ---
        shop_tile_count = len(self.shop_tiles)
        self.shop_tiles = flat_state[idx:idx + shop_tile_count].tolist()
        idx += shop_tile_count

        # --- Cat tiles ---
        cat_tile_shape = np.shape(self.cat_tiles)
        cat_tile_count = np.prod(cat_tile_shape)
        self.cat_tiles = flat_state[idx:idx + cat_tile_count].reshape(cat_tile_shape)
        idx += cat_tile_count

        # --- Board matrix ---
        board_shape = np.shape(self.board_matrix)
        board_size = np.prod(board_shape)
        self.board_matrix = flat_state[idx:idx + board_size].reshape(board_shape)

    def __str__(self):
        s = []
        s.append("=== Calico Environment ===")

        # Tile pool summary
        if isinstance(self.tile_pool, np.ndarray):
            s.append(f"Tile Pool: {self.tile_pool.tolist()}")
        else:
            s.append(f"Tile Pool: {self.tile_pool}")

        # Player and shop info
        s.append(f"Player Tiles: {self.player_tiles}")
        s.append(f"Shop Tiles: {self.shop_tiles}")

        # Cat tiles
        if isinstance(self.cat_tiles, np.ndarray):
            s.append(f"Cat Tiles: {self.cat_tiles.tolist()}")
        else:
            s.append(f"Cat Tiles: {self.cat_tiles}")

        # Board visualization
        s.append("Board Matrix:")
        if isinstance(self.board_matrix, np.ndarray):
            board_str = "\n".join(
                ["  " + " ".join(f"{x:3}" for x in row) for row in self.board_matrix]
            )
            s.append(board_str)
        else:
            s.append(str(self.board_matrix))

        return "\n".join(s)

env = CalicoEnv()
env.start_game()

print(env)