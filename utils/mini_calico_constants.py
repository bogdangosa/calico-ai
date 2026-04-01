# --- MINI CALICO CONSTANTS ---
# Reduced complexity for debugging and initial learning
BOARD_SIZE = 5  # 5x5 Matrix (1-tile border, 3x3 playable area)
TILE_COLORS = 3  # Reduced from 6
TILE_PATTERNS = 3  # Reduced from 6
TOTAL_TILE_TYPES = TILE_COLORS * TILE_PATTERNS  # 9 unique tiles
NR_OF_IDENTICAL_TILES = 4  # 9 * 4 = 40 tiles total
NR_OF_TILES_IN_SHOP = 3
PLAYER_HAND_SIZE = 2
NO_TILE_VALUE = -1  # Using -1 for empty is safer for CNNs than 37
OBJECTIVE_VALUE_BASE = -100  # Objectives are negative
MIN_REGION_FOR_SCORING = 3

# Simple Objective in the center (Index 2,2)
OBJECTIVE_POSITIONS_ON_BOARD = [[2, 2]]