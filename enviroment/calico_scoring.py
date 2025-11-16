import numpy as np
from utils.constants import *

# -------------------------
# Tile helpers
# -------------------------
def get_color_id(tile_id):
    return tile_id // TILE_COLORS

def get_pattern_id(tile_id):
    return tile_id % TILE_PATTERNS

def are_tiles_same_color(tile_id1, tile_id2):
    return get_color_id(tile_id1) == get_color_id(tile_id2)

def are_tiles_same_pattern(tile_id1, tile_id2):
    return get_pattern_id(tile_id1) == get_pattern_id(tile_id2)

# -------------------------
# Neighbours (hex-style)
# -------------------------
def get_neighbours(y, x):
    directions = [[0, 1], [0, -1], [1, 0], [-1, 0]]
    if y % 2 == 0:
        directions += [[-1, -1], [1, -1]]
    else:
        directions += [[-1, 1], [1, 1]]

    neighbours = []
    for dy, dx in directions:
        ny, nx = y + dy, x + dx
        if 0 <= ny < BOARD_SIZE and 0 <= nx < BOARD_SIZE:
            neighbours.append((ny, nx))
    return neighbours

# -------------------------
# Board printing
# -------------------------
def print_board(board_matrix):
    for row in board_matrix:
        print(" ".join(f"{v // TILE_COLORS}" for v in row))
    print()
    for row in board_matrix:
        print(" ".join(str(v) for v in row))
    print()

# -------------------------
# Flood fill for color/pattern regions
# -------------------------
def flood_fill(board_matrix, y, x, visited, get_id):
    start_tile = board_matrix[y][x]
    color_id = get_id(start_tile)
    stack = [(y, x)]
    region = []

    while stack:
        py, px = stack.pop()
        key = f"{py},{px}"
        if key in visited:
            continue
        visited[key] = True
        region.append((py, px))

        for ny, nx in get_neighbours(py, px):
            neighbor_key = f"{ny},{nx}"
            if neighbor_key in visited:
                continue
            neighbor_tile = board_matrix[ny][nx]
            if neighbor_tile < 0 or neighbor_tile == NO_TILE_VALUE:
                continue
            if get_id(neighbor_tile) == color_id:
                stack.append((ny, nx))
    return region

# -------------------------
# Score helpers
# -------------------------
def create_occurances_map(board_matrix, neighbours, get_id):
    occ_map = {}
    for ny, nx in neighbours:
        tile_id = board_matrix[ny][nx]
        used_id = get_id(tile_id)
        occ_map[used_id] = occ_map.get(used_id, 0) + 1
    return occ_map

def is_objective_full(board_matrix, neighbours):
    for ny, nx in neighbours:
        if board_matrix[ny][nx] == NO_TILE_VALUE:
            return False
    return True

def are_all_different(occ_map):
    return all(v == 1 for v in occ_map.values())

def has_two_groups_of_three(occ_map):
    counts = sorted(occ_map.values(), reverse=True)
    return len(counts) >= 2 and counts[0] >= 3 and counts[1] >= 3

def has_three_groups_of_two(occ_map):
    counts = sorted(occ_map.values(), reverse=True)
    return len(counts) >= 3 and counts[0] >= 2 and counts[1] >= 2 and counts[2] >= 2

# -------------------------
# Objective scoring
# -------------------------
def calculate_objective_1(colors_map, pattern_map):
    all_colors = are_all_different(colors_map)
    all_patterns = are_all_different(pattern_map)
    if all_colors and all_patterns:
        return 15
    if all_colors or all_patterns:
        return 10
    return 0

def calculate_objective_2(colors_map, pattern_map):
    colors_2_2_2 = has_three_groups_of_two(colors_map)
    patterns_2_2_2 = has_three_groups_of_two(pattern_map)
    if colors_2_2_2 and patterns_2_2_2:
        return 11
    if colors_2_2_2 or patterns_2_2_2:
        return 7
    return 0

def calculate_objective_3(colors_map, pattern_map):
    colors_3_3 = has_two_groups_of_three(colors_map)
    patterns_3_3 = has_two_groups_of_three(pattern_map)
    if colors_3_3 and patterns_3_3:
        return 13
    if colors_3_3 or patterns_3_3:
        return 8
    return 0

def calculate_objective_score(board_matrix, neighbours, obj_type):
    if not is_objective_full(board_matrix, neighbours):
        return 0
    colors_map = create_occurances_map(board_matrix, neighbours, get_color_id)
    patterns_map = create_occurances_map(board_matrix, neighbours, get_pattern_id)
    if obj_type == -1:
        return calculate_objective_1(colors_map, patterns_map)
    if obj_type == -2:
        return calculate_objective_2(colors_map, patterns_map)
    if obj_type == -3:
        return calculate_objective_3(colors_map, patterns_map)
    return 0

def get_objectives_score_on_board(board_matrix):
    score = 0
    for y, x in OBJECTIVE_POSITIONS_ON_BOARD:
        obj_type = board_matrix[y][x]
        neighbours = get_neighbours(y, x)
        score += calculate_objective_score(board_matrix, neighbours, obj_type)
    return score

def get_color_score_on_board(board_matrix):
    visited = {}
    score = 0
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
            region = flood_fill(board_matrix, y, x, visited, get_color_id)
            if len(region) >= MIN_REGION_FOR_SCORING:
                score += 3
    return score

def get_cats_score_on_board(board_matrix, cat_tiles):
    visited = {}
    score = 0
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
            if len(region) >= 3:
                tile_pattern = get_pattern_id(tile_value)
                try:
                    cat_pattern_index = list(cat_tiles).index(tile_pattern + 1)
                except ValueError:
                    continue
                if cat_pattern_index < 2:
                    score += 3
                elif cat_pattern_index < 4 and len(region) >= 4:
                    score += 5
                elif cat_pattern_index < 6 and len(region) >= 5:
                    score += 7
    return score

def get_total_score_on_board(board_matrix, cat_tiles):
    cats_score = get_cats_score_on_board(board_matrix, cat_tiles)
    color_score = get_color_score_on_board(board_matrix)
    objectives_score = get_objectives_score_on_board(board_matrix)
    total_score = cats_score + color_score + objectives_score
    return total_score

def get_total_score_on_board_detailed(board_matrix, cat_tiles):
    cats_score = get_cats_score_on_board(board_matrix, cat_tiles)
    color_score = get_color_score_on_board(board_matrix)
    objectives_score = get_objectives_score_on_board(board_matrix)
    total_score = cats_score + color_score + objectives_score
    return total_score,cats_score,color_score,objectives_score