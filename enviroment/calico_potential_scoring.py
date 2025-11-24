import random

from enviroment.calico_scoring import flood_fill, get_pattern_id, get_color_id, get_neighbours, get_total_score_on_board
from utils.constants import *
from collections import Counter
import math

def cat_potential_score(board_matrix, cat_tiles):
    """
    Estimate potential score for cat patterns.
    Mimics get_cats_score_on_board but gives heuristic value
    for regions not yet fully scoring.
    """
    score = 0
    visited = {}
    for y, x in OBJECTIVE_POSITIONS_ON_BOARD:
        visited[f"{y},{x}"] = True

    for y in range(BOARD_SIZE):
        for x in range(BOARD_SIZE):
            key = f"{y},{x}"
            if key in visited:
                continue
            tile_value = board_matrix[y][x]
            if tile_value < 0 or tile_value == NO_TILE_VALUE:
                visited[key] = True
                continue

            region = flood_fill(board_matrix, y, x, visited, get_pattern_id)
            region_size = len(region)
            tile_pattern = get_pattern_id(tile_value)
            try:
                cat_pattern_index = list(cat_tiles).index(tile_pattern + 1)
            except ValueError:
                continue

            # Heuristic potential scoring
            if region_size >= 3:
                if cat_pattern_index < 2:
                    score += 3
                elif cat_pattern_index < 4 and region_size >= 4:
                    score += 5
                elif cat_pattern_index < 6 and region_size >= 5:
                    score += 7
            else:
                # Give small potential for smaller regions
                if region_size == 2:
                    score += 1.5
                elif region_size == 1:
                    score += 0.5

    return score

def color_potential_score(board):
    score = 0
    visited = set()
    for y in range(BOARD_SIZE):
        for x in range(BOARD_SIZE):
            if (y,x) in visited: continue
            tile = board[y][x]
            if tile < 0 or tile == NO_TILE_VALUE:
                continue
            region = flood_fill(board, y, x, {}, get_color_id)
            for cell in region:
                visited.add(cell)

            size = len(region)

            if size == 1:
                score += 0
            elif size == 2:
                score += 3
            elif size >= 3:
                score += 5
    return score

def count_occ(lst):
    """
    Count occurrences of each unique element in a list and return counts as a list.
    Equivalent to using occ_map.values() in your other functions.
    """
    from collections import Counter
    return list(Counter(lst).values())

# helper wrappers (adapt if your actual functions are named differently)
# - get_neighbours(y,x) -> list of (ny,nx)
# - get_color_id(tile_id), get_pattern_id(tile_id)
# - NO_TILE_VALUE constant exists

def _counts_nonneg(items):
    """Return Counter over non-negative tile ids (skip NO_TILE_VALUE / None)."""
    return Counter([t for t in items if t is not None and t != NO_TILE_VALUE and t >= 0])

def _max_possible_groups_from_counts(counts: Counter, empty_slots: int, group_size: int):
    """
    Greedy estimation of how many full groups of size `group_size` we can form given:
      - counts: Counter of existing same-value frequencies (e.g., color frequencies)
      - empty_slots: how many cells we can still fill
    Returns integer number of groups achievable.
    """
    # Start with existing complete groups
    groups = 0
    # How many full groups we can form out of existing counts without changing tiles:
    for v in counts.values():
        groups += v // group_size

    # Remaining items that can be augmented with empty slots to form more groups:
    remainders = []
    for v in counts.values():
        r = v % group_size
        if r > 0:
            # need (group_size - r) empties to complete one more group of this value
            remainders.append(group_size - r)
    # also we can start groups from scratch using group_size empties (if no existing value)
    remainders.sort()  # use smallest costs first
    for cost in remainders:
        if empty_slots >= cost:
            empty_slots -= cost
            groups += 1
        else:
            break
    # Finally, any leftover empty_slots can create new groups from scratch:
    groups += empty_slots // group_size
    return groups

def objective_viability(board, tile_pool=None):
    """
    Returns a scalar score (sum of per-objective viability).
    Per-objective viability is in [0,1]. You can normalize later.
    If tile_pool (array or Counter) is provided, it is used to refine probabilities.
    """
    total_score = 0.0

    for (oy, ox) in OBJECTIVE_POSITIONS_ON_BOARD:
        obj_type = board[oy][ox]
        neighbours = get_neighbours(oy, ox)  # list of (y,x)
        # collect neighbor tiles (only playable tile ids >=0); leaving NO_TILE_VALUE or negatives out
        neighbor_tiles = [board[y][x] for (y, x) in neighbours if board[y][x] is not None and board[y][x] != NO_TILE_VALUE and board[y][x] >= 0]
        total_slots = len(neighbours)
        filled = len(neighbor_tiles)
        empty_slots = total_slots - filled

        # quick-fail if weird
        if total_slots == 0:
            continue

        # build color/pattern lists and counters
        colors = [get_color_id(t) for t in neighbor_tiles]
        patterns = [get_pattern_id(t) for t in neighbor_tiles]
        color_cnt = Counter(colors)
        pattern_cnt = Counter(patterns)

        # base weight: how much the objective matters depends on how filled it is -
        # more filled = more informative. weight in [0.6..1.2] (tweakable)
        base_weight = 0.5 + (filled / float(total_slots))  # between 0.5 and 1.5

        # compute viability per objective type
        viability = 0.0

        if obj_type == -1:
            # ALL-DIFFERENT objective: need all neighbor colors (and patterns) to be unique.
            needed_unique = total_slots

            # uniqueness possible check for colors
            unique_colors_now = len(set(colors))
            max_unique_with_fill = unique_colors_now + empty_slots
            color_possible = max_unique_with_fill >= needed_unique
            if not color_possible:
                color_viability = 0.0
            else:
                # fractional progress toward uniqueness
                color_viability = min(1.0, (unique_colors_now + empty_slots) / float(needed_unique))

            # patterns similarly
            unique_patterns_now = len(set(patterns))
            max_unique_pat = unique_patterns_now + empty_slots
            pattern_viability = 0.0 if max_unique_pat < needed_unique else min(1.0, (unique_patterns_now + empty_slots) / float(needed_unique))

            # conservative: both color and pattern must hold -> take product,
            # but blend so single strong signal still counts
            viability = 0.6 * (color_viability * pattern_viability) + 0.4 * ((color_viability + pattern_viability) / 2.0)

        elif obj_type == -3:
            # TWO GROUPS OF THREE (assume total_slots typically 6)
            group_size = 3
            needed_groups = 2
            # color-based grouping potential
            color_groups = _max_possible_groups_from_counts(color_cnt, empty_slots, group_size)
            color_viability = min(1.0, color_groups / float(needed_groups))
            # pattern-based
            pattern_groups = _max_possible_groups_from_counts(pattern_cnt, empty_slots, group_size)
            pattern_viability = min(1.0, pattern_groups / float(needed_groups))
            # objective satisfied if either colors or patterns form required groups -> take max
            viability = max(color_viability, pattern_viability)

        elif obj_type == -2:
            # THREE GROUPS OF TWO (assume total_slots typically 6)
            group_size = 2
            needed_groups = 3
            color_groups = _max_possible_groups_from_counts(color_cnt, empty_slots, group_size)
            color_viability = min(1.0, color_groups / float(needed_groups))
            pattern_groups = _max_possible_groups_from_counts(pattern_cnt, empty_slots, group_size)
            pattern_viability = min(1.0, pattern_groups / float(needed_groups))
            viability = max(color_viability, pattern_viability)

        else:
            # unknown objective; fallback: reward filled ratio
            viability = filled / float(total_slots)

        # optional: if tile_pool is provided, refine the viability estimate
        # (very simple heuristic: if tile_pool lacks enough distinct colors/patterns, reduce viability)
        if tile_pool is not None:
            # tile_pool assumed array-like counts indexed by tile_id; convert to color/pattern availability
            # Build color availability counts
            color_avail = Counter()
            pattern_avail = Counter()
            # iterate possible tile ids present in pool (tile_pool can be np.array of counts)
            for tid, cnt in enumerate(tile_pool):
                if cnt <= 0:
                    continue
                c = get_color_id(tid)
                p = get_pattern_id(tid)
                color_avail[c] += cnt
                pattern_avail[p] += cnt

            # if we need many unique colors but availability is low, scale down
            # required_unique_colors = max(0, needed_unique - len(set(colors)))
            # approximate available unique colors
            avail_unique_colors = sum(1 for v in color_avail if color_avail[v] > 0)
            # scale factor between 0..1
            if obj_type == -1:
                need = needed_unique
                if avail_unique_colors + len(set(colors)) < need:
                    # impossible given tile pool
                    viability *= 0.0
                else:
                    # small boost if there is plenty of unique colors
                    viability *= min(1.0, (avail_unique_colors + len(set(colors))) / float(need))

            # for grouping objectives, check total tiles available to form groups
            if obj_type in (-2, -3):
                # rough available groups for colors
                total_color_tiles_available = sum(color_avail.values()) + sum(color_cnt.values())
                # if not enough tiles exist at all, set to zero
                if total_color_tiles_available < (group_size * needed_groups):
                    # impossible by pool counts (very strict)
                    viability *= 0.0
                # otherwise keep viability as computed

        # combine weight
        total_score += base_weight * viability

    return total_score
def objective_viability_old(board):
    score = 0
    for (oy, ox) in OBJECTIVE_POSITIONS_ON_BOARD:
        obj_type = board[oy][ox]
        neighbours = get_neighbours(oy, ox)

        tiles = [board[y][x] for (y,x) in neighbours if board[y][x] >= 0]

        # reward if placing tile improves diversity or grouping
        colors = [get_color_id(t) for t in tiles]
        patterns = [get_pattern_id(t) for t in tiles]

        # the more filled the better
        filled = len(tiles)
        score += filled * 2.5

        # diversity or grouping potential
        if obj_type == -1:  # all-different objective
            score += len(set(colors)) * 2
            score += len(set(patterns)) * 2

        elif obj_type == -3:  # two groups of three possibility
            score += max(count_occ(colors)) * 1.8 if colors else 0
            score += max(count_occ(patterns)) * 1.8 if patterns else 0

        elif obj_type == -2:  # three groups of two possibility
            score += sum(count_occ(colors)) * 0.8 if colors else 0
            score += sum(count_occ(patterns)) * 0.8 if patterns else 0
    return score

def generate_random_config(min_val=0.0, max_val=5.0):
    return {
        "WEIGHT_FINAL_SCORE": random.uniform(min_val, max_val),
        "WEIGHT_CAT_POTENTIAL": random.uniform(min_val, max_val),
        "WEIGHT_COLOR_POTENTIAL": random.uniform(min_val, max_val),
        "WEIGHT_OBJECTIVE_VIABILITY": random.uniform(min_val, max_val),
    }

def evaluate_move(board_matrix, cat_tiles,config=EVALUATION_CONFIG):
    return (
        config["WEIGHT_FINAL_SCORE"] * get_total_score_on_board(board_matrix, cat_tiles) +
        config["WEIGHT_CAT_POTENTIAL"] * cat_potential_score(board_matrix, cat_tiles) +
        config["WEIGHT_COLOR_POTENTIAL"] * color_potential_score(board_matrix) +
        config["WEIGHT_OBJECTIVE_VIABILITY"] * objective_viability(board_matrix)
    )

numbers = [
    56, 73, 57, 64, 77, 52, 51, 73, 51, 48, 61, 60, 72, 49, 64, 64, 53,
    57, 67, 64, 49, 41, 57, 47, 35, 73, 78, 73, 62, 71, 42, 53, 60, 45, 54,
    68, 61
]

average = sum(numbers) / len(numbers)
print(average)
