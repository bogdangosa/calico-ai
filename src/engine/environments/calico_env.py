import copy
import json
import logging
import numpy as np

from src.engine.environments.history_manager import HistoryManager
from src.engine.scoring.scoring import ScoringCalculator
from src.models.game_config import GameSettings
from src.models.game_models import CalicoAction, ActionType, BoardColors


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


    def disable_history(self):
        self.history_manager = None

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

    def start_game(self,shuffle_cat_tiles=False,board_color=BoardColors.PURPLE,seed=66):
        self.tile_pool = self.initiate_tile_pool()
        self.player_tiles = [self.generate_random_tile() for _ in range(self.config.player_hand_size)]
        self.shop_tiles = self.initiate_shop_tiles()
        self.cat_tiles = self.initialize_cat_tiles(shuffle_cat_tiles)
        self.board_matrix = self.initialize_inner_board()
        self.initialize_outer_board(board_color)
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

    def initialize_cat_tiles(self,shuffle_tiles=False):
        cat_tiles = np.arange(0, self.config.tiles.cat_types)
        if shuffle_tiles:
            np.random.shuffle(cat_tiles)  # shuffles in place
        return cat_tiles

    def generate_random_tile(self):
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
        board_matrix = np.full((self.size, self.size), self.config.board.no_tile_value, dtype=int)

        for i, (row, col) in enumerate(self.config.board.objective_positions, start=1):
            board_matrix[row, col] = -i

        return board_matrix

    def initialize_outer_board(self, board_color: str):
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
