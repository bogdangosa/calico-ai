import numpy as np
import logging
from typing import List, Optional

from src.models.game_config import GameSettings
from src.engine.environments.history_manager import HistoryManager


class CalicoPlayerEnv:
    """Handles the private state and logic for a single player's board and hand."""

    def __init__(self, player_id: int, config: GameSettings):
        self.player_id = player_id
        self.config = config
        self.size = config.board.size
        self.board_matrix: np.ndarray = np.zeros((self.size, self.size), dtype=int)
        self.player_tiles: List[int] = []
        self.selected_player_tile_index: int = 0
        self.history_manager: Optional[HistoryManager] = None

    def initialize_player(self, initial_tiles: List[int], board_color: str = "purple"):
        self.player_tiles = initial_tiles
        self.board_matrix = self._initialize_inner_board()
        self._initialize_outer_board(board_color)
        self.selected_player_tile_index = 0

    def _initialize_inner_board(self) -> np.ndarray:
        board_matrix = np.full((self.size, self.size), self.config.board.no_tile_value, dtype=int)
        for i, (row, col) in enumerate(self.config.board.objective_positions, start=1):
            board_matrix[row, col] = -i
        return board_matrix

    def _initialize_outer_board(self, board_color: str):
        border_tiles = self.config.board.borders.get(board_color)
        if not border_tiles:
            # Fallback to first available color if requested color doesn't exist
            colors = list(self.config.board.borders.keys())
            border_tiles = self.config.board.borders.get(colors[0])

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
            logging.warning(f"Border tile sequence for player {self.player_id} is too short.")

    def place_tile(self, row: int, col: int, tile_index: int):
        tile_id = self.player_tiles[tile_index]
        self.board_matrix[row][col] = tile_id
        self.player_tiles[tile_index] = self.config.board.no_tile_value
        self.selected_player_tile_index = tile_index

    def is_board_full(self) -> bool:
        return not np.any(self.board_matrix == self.config.board.no_tile_value)
