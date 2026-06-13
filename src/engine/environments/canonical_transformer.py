import numpy as np
from src.models.game_config import GameSettings

class CanonicalTransformer:
    """
    Transforms the game board and other state components into a canonical form.
    Mappings are built dynamically based on the order of appearance on the board.
    """
    def __init__(self, config: GameSettings):
        self.config = config
        self.num_colors = config.tiles.colors
        self.num_patterns = config.tiles.patterns
        self.empty_val = config.board.no_tile_value
        
        # Internal state to store mappings built during board transformation
        self.color_map = {}
        self.pattern_map = {}
        self.next_color_id = 0
        self.next_pattern_id = 0

    def _reset_maps(self):
        self.color_map = {}
        self.pattern_map = {}
        self.next_color_id = 0
        self.next_pattern_id = 0

    def transform_board(self, board: np.ndarray) -> np.ndarray:
        """
        Transforms the board and builds both color and pattern mappings.
        Colors and patterns are mapped by order of appearance (top-left to bottom-right).
        """
        self._reset_maps()
        size = board.shape[0]
        canonical_board = board.copy()
        
        for r in range(size):
            for c in range(size):
                tile_id = board[r, c]
                if self._is_standard_tile(tile_id):
                    original_color = tile_id // self.num_patterns
                    original_pattern = tile_id % self.num_patterns
                    
                    # Map Color (updates internal state)
                    if original_color not in self.color_map:
                        self.color_map[original_color] = self.next_color_id
                        self.next_color_id += 1
                    
                    # Map Pattern (updates internal state)
                    if original_pattern not in self.pattern_map:
                        self.pattern_map[original_pattern] = self.next_pattern_id
                        self.next_pattern_id += 1
                    
                    canonical_color = self.color_map[original_color]
                    canonical_pattern = self.pattern_map[original_pattern]
                    
                    canonical_board[r, c] = (canonical_color * self.num_patterns) + canonical_pattern
                    
        return canonical_board

    def transform_board_colors(self, board: np.ndarray) -> np.ndarray:
        """
        Transforms the board and builds ONLY the color mapping.
        """
        self._reset_maps()
        size = board.shape[0]
        canonical_board = board.copy()
        
        for r in range(size):
            for c in range(size):
                tile_id = board[r, c]
                if self._is_standard_tile(tile_id):
                    original_color = tile_id // self.num_patterns
                    original_pattern = tile_id % self.num_patterns
                    
                    if original_color not in self.color_map:
                        self.color_map[original_color] = self.next_color_id
                        self.next_color_id += 1
                    
                    canonical_color = self.color_map[original_color]
                    canonical_board[r, c] = (canonical_color * self.num_patterns) + original_pattern
                    
        return canonical_board

    def transform_patterns(self, patterns: list) -> list:
        """
        Transforms a list of pattern IDs (e.g., cat_tiles) using the current mapping.
        If a pattern was not on the board, it is assigned the next available canonical ID.
        """
        canonical_patterns = []
        for p in patterns:
            if p not in self.pattern_map:
                self.pattern_map[p] = self.next_pattern_id
                self.next_pattern_id += 1
            canonical_patterns.append(self.pattern_map[p])
        
        # Return as the same type as input (preserving list or numpy array)
        if isinstance(patterns, np.ndarray):
            return np.array(canonical_patterns)
        return canonical_patterns

    def transform_colors(self, colors: list) -> list:
        """
        Transforms a list of color IDs using the current mapping.
        """
        canonical_colors = []
        for c in colors:
            if c not in self.color_map:
                self.color_map[c] = self.next_color_id
                self.next_color_id += 1
            canonical_colors.append(self.color_map[c])
            
        if isinstance(colors, np.ndarray):
            return np.array(canonical_colors)
        return canonical_colors

    def get_inner_board(self, board: np.ndarray) -> np.ndarray:
        """
        Returns only the inner playable part of the board, removing the static borders.
        For a board of size NxN, it returns a matrix of size (N-2)x(N-2).
        """
        if board.shape[0] <= 2:
            return board
        return board[1:-1, 1:-1]

    def _is_standard_tile(self, tile_id: int) -> bool:
        return tile_id >= 0 and tile_id != self.empty_val
