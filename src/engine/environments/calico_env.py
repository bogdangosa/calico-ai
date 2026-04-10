import copy
import json
import logging
import numpy as np

from src.engine.environments.history_manager import HistoryManager
from src.engine.scoring.scoring import ScoringCalculator
from src.models.game_config import GameSettings
from src.models.game_models import CalicoAction, ActionType


class CalicoEnv:
    def __init__(self,config: GameSettings):
        self.config = config
        self.size = config.board.size
        self.history_manager = None
        
        self.tile_pool = None
        self.player_tiles = []
        self.shop_tiles = []
        self.cat_tiles = []
        self.board_matrix = []
        self.move_history = []
        self.mode = ""
        self.selected_player_tile_index = 0

    def enable_history(self):
        self.history_manager = HistoryManager(self)

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


    def get_legal_actions(self)->list[CalicoAction]:
        legal_actions = []
        if self.mode == ActionType.PLACE:
            empty_slots = [
                (r, c) for r in range(self.size) for c in range(self.size)
                if self.board_matrix[r][c] == self.config.board.no_tile_value
            ]

            for idx, tile_id in enumerate(self.player_tiles):
                if tile_id == self.config.board.no_tile_value:
                    continue

                for r, c in empty_slots:
                    legal_actions.append(CalicoAction(
                        action_type=ActionType.PLACE,
                        tile_index=idx,
                        row=r,
                        col=c
                    ))

        elif self.mode == ActionType.BUY:
            for shop_idx in range(len(self.shop_tiles)):
                legal_actions.append(CalicoAction(
                    action_type=ActionType.BUY,
                    tile_index=shop_idx
                ))

        return legal_actions

    def perform_action(self, action: CalicoAction):
        if self.history_manager:
            self.history_manager.record_and_perform(action)
        elif action.action_type == ActionType.PLACE:
            self.place_tile(action.row, action.col, action.tile_index)
        elif action.action_type == ActionType.BUY:
            self.buy_tile(action.tile_index)

    def set_selected_from_empty(self):
        for index,tile in enumerate(self.player_tiles, start=0):
            if tile == self.config.board.no_tile_value:
                self.selected_player_tile_index = index

    def undo_action(self):
        if self.history_manager:
            self.history_manager.undo()
        else:
            raise RuntimeError("History manager not initialized!")

    def start_game(self,seed=41):
        self.tile_pool = self.initiate_tile_pool()
        self.player_tiles = [self.generate_random_tile() for _ in range(self.config.player_hand_size)]
        self.shop_tiles = self.initiate_shop_tiles()
        self.cat_tiles = self.initialize_cat_tiles()
        self.board_matrix = self.initialize_inner_board()
        self.initialize_outer_board("purple")
        self.mode = ActionType.PLACE
        self.selected_player_tile_index = 0

    def is_game_over(self):
        return not np.any(self.board_matrix == self.config.board.no_tile_value)

    def buy_tile(self, tile_index: int):
        """Buy a tile from the shop and update the player hand."""
        self.mode = ActionType.PLACE
        bought_tile_id = self.shop_tiles[tile_index]
        self.player_tiles[self.selected_player_tile_index] = bought_tile_id
        self.replace_tile(tile_index)
        self.replace_tile(0)
        return bought_tile_id

    def place_tile(self, row: int, col: int, selected_tile_index: int):
        """Place a selected tile on the board and update score."""
        selected_tile_id = self.player_tiles[selected_tile_index]
        self.board_matrix[row][col] = selected_tile_id
        self.player_tiles[selected_tile_index] = self.config.board.no_tile_value
        self.selected_player_tile_index = selected_tile_index
        self.mode = ActionType.BUY

    def replace_tile(self, tile_index: int):
        """
        Replace a tile in the shop at `tile_index` with a new random tile.
        """
        self.shop_tiles = np.delete(self.shop_tiles, tile_index)

        new_tile = self.generate_random_tile()
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

        if isinstance(self.tile_pool, np.ndarray):
            s.append(f"Tile Pool: {self.tile_pool.tolist()}")
        else:
            s.append(f"Tile Pool: {self.tile_pool}")

        s.append(f"Player Tiles: {self.player_tiles}")
        s.append(f"Shop Tiles: {self.shop_tiles}")

        if isinstance(self.cat_tiles, np.ndarray):
            s.append(f"Cat Tiles: {self.cat_tiles.tolist()}")
        else:
            s.append(f"Cat Tiles: {self.cat_tiles}")

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
