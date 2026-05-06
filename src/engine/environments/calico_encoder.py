import numpy as np


class CalicoEncoder:
    def __init__(self, config):
        self.config = config
        self.colors = config.tiles.colors
        self.patterns = config.tiles.patterns
        self.total_feature_layers = self.colors + self.patterns + 1

    def encode(self, env):
        board = env.board_matrix
        size = env.size
        tensor = np.zeros((size, size, self.total_feature_layers), dtype=np.float32)

        for r in range(size):
            for c in range(size):
                tile_id = board[r, c]
                if self._is_standard_tile(tile_id):
                    self._map_tile_features(tensor, r, c, tile_id)
                elif self._is_objective_tile(tile_id):
                    self._map_objective_layer(tensor, r, c)

        return tensor

    def _is_standard_tile(self, tile_id):
        return tile_id >= 0

    def _is_objective_tile(self, tile_id):
        return tile_id < 0 and tile_id != self.config.board.no_tile_value

    def _map_tile_features(self, tensor, r, c, tile_id):
        color_idx = tile_id // self.patterns
        pattern_idx = tile_id % self.patterns

        tensor[r, c, color_idx] = 1.0
        tensor[r, c, self.colors + pattern_idx] = 1.0

    def _map_objective_layer(self, tensor, r, c):
        tensor[r, c, self.colors + self.patterns] = 1.0