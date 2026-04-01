import numpy as np


class ScoringCalculator:
    def __init__(self, config):
        self.config = config
        self.size = config.board.size
        self.empty_val = config.board.no_tile_value

    def get_color_id(self, tile_id: int) -> int:
        return tile_id // self.config.tiles.colors

    def get_pattern_id(self, tile_id: int) -> int:
        return tile_id % self.config.tiles.patterns

    def _get_occurrence_map(self, board, neighbors, id_func):
        occ_map = {}
        for r, c in neighbors:
            tile_id = board[r, c]
            attr_id = id_func(tile_id)
            occ_map[attr_id] = occ_map.get(attr_id, 0) + 1
        return occ_map

    def _has_all_different(self, occ_map):
        return all(count == 1 for count in occ_map.values()) and len(occ_map) == 6

    def _has_three_pairs(self, occ_map):
        # Must have at least 3 distinct IDs, each with at least 2 tiles
        pairs = [count for count in occ_map.values() if count >= 2]
        return len(pairs) >= 3

    def _has_two_triplets(self, occ_map):
        triplets = [count for count in occ_map.values() if count >= 3]
        return len(triplets) >= 2

    def get_neighbors(self, row: int, col: int):
        """Returns valid neighbor coordinates for a staggered hex-grid."""
        directions = [[0, 1], [0, -1], [1, 0], [-1, 0]]

        if row % 2 == 0:
            directions += [[-1, -1], [1, -1]]
        else:
            directions += [[-1, 1], [1, 1]]

        neighbors = []
        for dr, dc in directions:
            nr, nc = row + dr, col + dc
            if 0 <= nr < self.size and 0 <= nc < self.size:
                neighbors.append((nr, nc))
        return neighbors

    def find_region(self, board, start_row, start_col, visited, attribute_func):
        """Flood fill to find connected tiles sharing the same attribute (color or pattern)."""
        target_tile = board[start_row, start_col]
        target_id = attribute_func(target_tile)

        stack = [(start_row, start_col)]
        region = []

        while stack:
            r, c = stack.pop()
            if (r, c) in visited:
                continue

            visited.add((r, c))
            region.append((r, c))

            for nr, nc in self.get_neighbors(r, c):
                neighbor_tile = board[nr, nc]
                if neighbor_tile < 0 or neighbor_tile == self.empty_val:
                    continue

                if attribute_func(neighbor_tile) == target_id:
                    stack.append((nr, nc))
        return region

    def get_color_score(self, board):
        """Scores 3 points for every group of same-colored tiles (min size in config)."""
        visited = set()
        score = 0
        min_size = self.config.min_region_for_scoring

        for r, c in self.config.board.objective_positions:
            visited.add((r, c))

        for r in range(self.size):
            for c in range(self.size):
                if (r, c) in visited or board[r, c] < 0 or board[r, c] == self.empty_val:
                    continue

                region = self.find_region(board, r, c, visited, self.get_color_id)
                if len(region) >= min_size:
                    score += 3
        return score


    def get_objective_score(self, board):
        score = 0
        for r, c in self.config.board.objective_positions:
            obj_type = board[r, c]
            neighbors = self.get_neighbors(r, c)

            if any(board[nr, nc] == self.empty_val for nr, nc in neighbors):
                continue

            colors_map = self._get_occurrence_map(board, neighbors, self.get_color_id)
            patterns_map = self._get_occurrence_map(board, neighbors, self.get_pattern_id)

            score += self._calculate_score_by_type(obj_type, colors_map, patterns_map)
        return score

    def _calculate_score_by_type(self, obj_type, colors_map, patterns_map):
        if obj_type == -1:
            match_c = self._has_all_different(colors_map)
            match_p = self._has_all_different(patterns_map)
            return 15 if (match_c and match_p) else (10 if match_c or match_p else 0)

        if obj_type == -2:
            match_c = self._has_three_pairs(colors_map)
            match_p = self._has_three_pairs(patterns_map)
            return 11 if (match_c and match_p) else (7 if match_c or match_p else 0)

        if obj_type == -3:
            match_c = self._has_two_triplets(colors_map)
            match_p = self._has_two_triplets(patterns_map)
            return 13 if (match_c and match_p) else (8 if match_c or match_p else 0)

        return 0

    def get_cat_score(self, board, cat_config_list):
        """
        Scores cats based on pattern regions.
        cat_config_list: A list of pattern IDs that correspond to active cats.
        """
        visited = set()
        total_cat_score = 0

        for r, c in self.config.board.objective_positions:
            visited.add((r, c))

        for r in range(self.size):
            for c in range(self.size):
                tile_val = board[r, c]

                if (r, c) in visited or tile_val < 0 or tile_val == self.empty_val:
                    continue

                region = self.find_region(board, r, c, visited, self.get_pattern_id)
                region_size = len(region)
                pattern_id = self.get_pattern_id(tile_val)

                if pattern_id in cat_config_list:
                    cat_index = cat_config_list.index(pattern_id)
                    total_cat_score += self._calculate_individual_cat_points(cat_index, region_size)

        return total_cat_score

    def _calculate_individual_cat_points(self, cat_index, region_size):
        """
        Maps the cat's difficulty/index to its scoring requirement.
        Based on standard Calico rules:
        - Cats 0-1: Group of 3 (3 pts)
        - Cats 2-3: Group of 4 (5 pts)
        - Cats 4-5: Group of 5 (7 pts)
        """
        if cat_index < 2:
            return 3 if region_size >= 3 else 0
        elif cat_index < 4:
            return 5 if region_size >= 4 else 0
        elif cat_index < 6:
            return 7 if region_size >= 5 else 0
        return 0

    def _get_counts(self, tile_ids, attribute_func):
        counts = {}
        for tid in tile_ids:
            attr = attribute_func(tid)
            counts[attr] = counts.get(attr, 0) + 1
        return sorted(counts.values(), reverse=True)

    def _normalize_cat_tiles(self, cat_tiles) -> list:
        """Ensures cat_tiles is a standard Python list for .index() compatibility."""
        if isinstance(cat_tiles, np.ndarray):
            return cat_tiles.tolist()

        if isinstance(cat_tiles, list):
            return cat_tiles

        try:
            return list(cat_tiles)
        except TypeError:
            return [cat_tiles] if cat_tiles is not None else []

    def get_total_detailed_score(self, board, cat_tiles):
        cat_tiles = self._normalize_cat_tiles(cat_tiles)

        color_pts = self.get_color_score(board)
        obj_pts = self.get_objective_score(board)
        cat_pts = self.get_cat_score(board, cat_tiles)

        total = color_pts + obj_pts + cat_pts
        return total, color_pts, obj_pts, cat_pts