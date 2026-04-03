from collections import Counter
from src.engine.scoring.scoring import ScoringCalculator


class PotentialScoringCalculator(ScoringCalculator):
    def __init__(self, config):
        super().__init__(config)
        self.weights = config.evaluation

    def evaluate_move(self, board, cat_tiles, tile_pool=None):
        """
        The main entry point for the Lookahead Agent.
        Combines actual scores with heuristic potential.
        """
        actual_total, actual_color, actual_obj, actual_cat = self.get_total_detailed_score(board, cat_tiles)

        pot_cat = self.get_cat_potential(board, cat_tiles)
        pot_color = self.get_color_potential(board)
        pot_obj = self.get_objective_viability(board, tile_pool)

        final_heuristic_score = (
                actual_total * self.weights.weight_final_score +
                pot_cat * self.weights.weight_cat_potential +
                pot_color * self.weights.weight_color_potential +
                pot_obj * self.weights.weight_objective_viability
        )

        return final_heuristic_score

    def get_cat_potential(self, board, cat_tiles):
        """Estimate potential for cat patterns in incomplete regions."""
        score = 0
        visited = set()
        cat_list = self._normalize_cat_tiles(cat_tiles)

        for r, c in self.config.board.objective_positions:
            visited.add((r, c))

        for r in range(self.size):
            for c in range(self.size):
                tile_val = board[r, c]
                if (r, c) in visited or tile_val < 0 or tile_val == self.empty_val:
                    continue

                region = self.find_region(board, r, c, visited, self.get_pattern_id)
                size = len(region)
                pattern_id = self.get_pattern_id(tile_val)

                if pattern_id in cat_list:
                    cat_idx = cat_list.index(pattern_id)
                    if size >= 3:
                        score += self._calculate_individual_cat_points(cat_idx, size)
                    elif size == 2:
                        score += 1.5
                    elif size == 1:
                        score += 0.5
        return score

    def get_color_potential(self, board):
        """Reward growing color regions before they hit the scoring threshold."""
        score = 0
        visited = set()
        min_scoring_size = self.config.min_region_for_scoring

        for r in range(self.size):
            for c in range(self.size):
                if (r, c) in visited or board[r, c] < 0 or board[r, c] == self.empty_val:
                    continue

                region = self.find_region(board, r, c, visited, self.get_color_id)
                size = len(region)

                if size >= min_scoring_size:
                    score += 5
                elif size == 2:
                    score += 3
                elif size == 1:
                    score += 0.5
        return score

    def get_objective_viability(self, board, tile_pool=None):
        """Calculates how likely an objective is to be completed based on empty slots."""
        total_viability = 0.0

        for r, c in self.config.board.objective_positions:
            obj_type = board[r, c]
            neighbors = self.get_neighbors(r, c)

            neighbor_tiles = [board[nr, nc] for nr, nc in neighbors if board[nr, nc] >= 0]
            total_slots = len(neighbors)
            empty_slots = total_slots - len(neighbor_tiles)

            if total_slots == 0: continue

            colors = [self.get_color_id(t) for t in neighbor_tiles]
            patterns = [self.get_pattern_id(t) for t in neighbor_tiles]

            base_weight = 0.5 + (len(neighbor_tiles) / total_slots)

            viability = 0.0
            if obj_type == -1:
                viability = self._check_all_diff_viability(colors, patterns, empty_slots, total_slots)
            elif obj_type == -2:
                viability = self._check_group_viability(colors, patterns, empty_slots, 2, 3)
            elif obj_type == -3:
                viability = self._check_group_viability(colors, patterns, empty_slots, 3, 2)

            total_viability += base_weight * viability

        return total_viability


    def _check_all_diff_viability(self, colors, patterns, empty, total):
        c_possible = (len(set(colors)) + empty) >= total
        p_possible = (len(set(patterns)) + empty) >= total

        if not c_possible and not p_possible: return 0.0

        c_prog = (len(set(colors)) + empty) / total
        p_prog = (len(set(patterns)) + empty) / total
        return 0.6 * (c_prog * p_prog) + 0.4 * ((c_prog + p_prog) / 2)

    def _check_group_viability(self, colors, patterns, empty, group_size, needed):
        c_groups = self._max_possible_groups(Counter(colors), empty, group_size)
        p_groups = self._max_possible_groups(Counter(patterns), empty, group_size)
        return max(min(1.0, c_groups / needed), min(1.0, p_groups / needed))

    def _max_possible_groups(self, counts, empty_slots, group_size):
        groups = sum(v // group_size for v in counts.values())
        remainders = sorted([group_size - (v % group_size) for v in counts.values() if v % group_size > 0])

        for cost in remainders:
            if empty_slots >= cost:
                empty_slots -= cost
                groups += 1
        groups += empty_slots // group_size
        return groups