import copy
import json
import random
import logging
import numpy as np

from src.engine.scoring import ScoringCalculator
from src.models.game_config import GameSettings

class CalicoEnv:
    def __init__(self,config: GameSettings):
        self.config = config
        self.size = config.board.size
        
        self.tile_pool = None
        self.player_tiles = []
        self.shop_tiles = []
        self.cat_tiles = []
        self.board_matrix = []
        self.move_history = []
        self.mode = ""
        self.selected_player_tile_index = 0

    def get_board_tensor(self):
        tensor = np.zeros((self.size, self.size, self.config.tiles.colors + self.config.tiles.patterns + 1))
        for r in range(self.size):
            for c in range(self.size):
                val = self.board_matrix[r, c]
                if val >= 0:
                    tensor[r, c, val // self.config.tiles.patterns] = 1.0
                    tensor[r, c, self.config.tiles.colors + (val % self.config.tiles.patterns)] = 1.0
                elif val < 0 and val != self.config.board.no_tile_value:
                    tensor[r, c, 12] = 1.0
        return tensor

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
                if tile_id == self.config.board.self.config.board.no_tile_value:
                    continue
                for r in range(self.size):
                    for c in range(self.size):
                        if self.board_matrix[r][c] == self.config.board.self.config.board.no_tile_value:
                            legal_actions.append(('place', idx, r, c))

        if self.mode == "buying":
            # Buy tile actions
            for shop_idx, shop_tile in enumerate(self.shop_tiles):
                legal_actions.append(('buy', shop_idx, None, None))

        return legal_actions

    def set_selected_from_empty(self):
        for index,tile in enumerate(self.player_tiles, start=0):
            if tile == self.config.board.no_tile_value:
                self.selected_player_tile_index = index

    def perform_action(self, action):
        """Apply an action to the environment and save it in move_history for undo."""
        action_type, tile_idx, row, col = action
        record = {"action_type": action_type}

        if action_type == 'place':
            record.update({
                "row": row,
                "col": col,
                "hand_index": tile_idx,
                "hand_value": self.player_tiles[tile_idx],
                "prev_tile": self.board_matrix[row][col],
                "prev_mode": self.mode,
                "prev_selected_index": self.selected_player_tile_index
            })
            self.place_tile(row, col, tile_idx)

        elif action_type == 'buy':
            record.update({
                "shop_index": tile_idx,
                "prev_mode": self.mode,
                "prev_selected_index": self.selected_player_tile_index,
                "hand_value": self.player_tiles[self.selected_player_tile_index],
                "shop_snapshot": copy.deepcopy(self.shop_tiles)
            })
            self.buy_tile(tile_idx)
            record.update({
                "new_shop_snapshot": copy.deepcopy(self.shop_tiles)
            })

        # Save the record for undo
        self.move_history.append(record)

    def undo_action(self):
        """Undo the last performed action."""
        if not self.move_history:
            return  # nothing to undo

        record = self.move_history.pop()
        action_type = record["action_type"]

        if action_type == "place":
            row = record["row"]
            col = record["col"]
            hand_index = record["hand_index"]

            # Restore board cell
            self.board_matrix[row][col] = record["prev_tile"]

            # Restore player hand
            self.player_tiles[hand_index] = record["hand_value"]

            # Restore mode and selected index
            self.mode = record["prev_mode"]
            self.selected_player_tile_index = record["prev_selected_index"]

        elif action_type == "buy":
            # Restore shop tiles exactly
            self.shop_tiles = record["shop_snapshot"].copy()

            # Restore player hand
            self.player_tiles[self.selected_player_tile_index] = record["hand_value"]

            # Restore the tile pool: return the purchased tile
            new_shop_snapshot = record["new_shop_snapshot"].copy()
            self.tile_pool[new_shop_snapshot[1]] += 1
            self.tile_pool[new_shop_snapshot[2]] += 1



            # Restore mode and selected index
            self.mode = record["prev_mode"]
            self.selected_player_tile_index = record["prev_selected_index"]

    def start_game(self,seed=41):
        self.tile_pool = self.initiate_tile_pool()
        self.player_tiles = [self.generate_random_tile() for _ in range(self.config.player_hand_size)]
        self.shop_tiles = self.initiate_shop_tiles()
        self.cat_tiles = self.initialize_cat_tiles()
        self.board_matrix = self.initialize_inner_board()
        self.initialize_outer_board("purple")
        self.mode = "placing"
        self.selected_player_tile_index = 0

    def is_game_over(self):
        return not np.any(self.board_matrix == self.config.board.no_tile_value)

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
        self.player_tiles[selected_tile_index] = self.config.board.no_tile_value
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
        return new_tile

    def initiate_tile_pool(self):
        return np.full(self.config.tiles.colors * self.config.tiles.patterns, self.config.tiles.identical_tiles)

    def initiate_shop_tiles(self):
        return [self.generate_random_tile() for _ in range(self.config.nr_of_tiles_in_shop)]

    def initiate_player_tiles(self):
        return [self.generate_random_tile() for _ in range(self.config.player_hand_size)]

    def initialize_cat_tiles(self):
        cat_tiles = np.arange(1, self.config.tiles.cat_types + 1)
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
        for i in range(self.size-2):
            for j in range(self.size-2):
                if [i+1,j+1] in self.config.board.objective_positions:
                    continue
                self.board_matrix[i+1][j+1] = self.generate_random_tile()

    def initialize_inner_board(self):
        """
        Creates a 5x5 playable board:
          - all cells initialized to self.config.board.no_tile_value
          - objectives placed at specified coordinates (negative IDs)
        """
        board_matrix = np.full((self.size, self.size), self.config.board.no_tile_value, dtype=int)

        # Place objectives (negative IDs)
        for i, (row, col) in enumerate(self.config.board.objective_positions, start=1):
            board_matrix[row, col] = -i

        return board_matrix

    def initialize_outer_board(self, board_color: str):
        """
        Populates the board perimeter with pre-defined border tiles.
        Uses a clockwise traversal: Top -> Right -> Bottom -> Left.
        """
        border_tiles = self.config.board.borders.get(board_color)
        if not border_tiles:
            raise ValueError(f"Border color '{board_color}' not found in configuration.")

        tile_iterator = iter(border_tiles)
        last_index = self.size - 1

        try:
            for col in range(self.size):
                self.board_matrix[0, col] = next(tile_iterator)

            for row in range(1, last_index):
                self.board_matrix[row, last_index] = next(tile_iterator)

            for col in range(last_index, -1, -1):
                self.board_matrix[last_index, col] = next(tile_iterator)

            for row in range(last_index - 1, 0, -1):
                self.board_matrix[row, 0] = next(tile_iterator)

        except StopIteration:

            logging.warning(
                f"Border tile sequence for '{board_color}' is shorter than "
                f"the required {4 * self.size - 4} tiles for a {self.size}x{self.size} board."
            )

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
        state_parts.append(np.array([int(self.mode == "buying")], dtype=int))
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
        self.mode = "buying" if flat_state[idx] == 1 else "placing"
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
        self.size = np.prod(board_shape)
        self.board_matrix = flat_state[idx:idx + self.size].reshape(board_shape)

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

if __name__ == "__main__":
    with open("../../../config/calico_settings.json", "r") as f:
        config_data = json.load(f)

    # 2. Parse into Pydantic model
    config_game = GameSettings(**config_data)
    env = CalicoEnv(config_game)
    env.start_game()

    env.fill_board_randomly()
    print(env)

    scoring = ScoringCalculator(config_game)

    score = scoring.get_total_detailed_score(env.board_matrix,env.cat_tiles)

    print(score)
