from enviroment.calico_scoring import flood_fill, get_pattern_id, get_color_id, get_neighbours, get_total_score_on_board
from utils.constants import *

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
                score += 1
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

def objective_viability(board):
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


def evaluate_move(board_matrix, cat_tiles):
    return (
        2 * get_total_score_on_board(board_matrix, cat_tiles) +
        2.5  * cat_potential_score(board_matrix,cat_tiles) +
        1.5  * color_potential_score(board_matrix) +
        3  * objective_viability(board_matrix)
    )

numbers = [
    56, 73, 57, 64, 77, 52, 51, 73, 51, 48, 61, 60, 72, 49, 64, 64, 53,
    57, 67, 64, 49, 41, 57, 47, 35, 73, 78, 73, 62, 71, 42, 53, 60, 45, 54,
    68, 61
]

average = sum(numbers) / len(numbers)
print(average)
