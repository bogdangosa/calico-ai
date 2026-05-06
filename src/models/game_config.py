from pydantic import BaseModel, Field
from typing import List, Dict


class BoardConfig(BaseModel):
    size: int = Field(2, description="The dimension of the square game board")
    no_tile_value: int = Field(0, description="The value representing an empty slot")
    objective_positions: List[List[int]] = Field(
        default=[[2, 3], [3, 4], [4, 2]],
        description="Coordinates for objective positions on the board"
    )
    borders: Dict[str, List[int]] = Field(
        default_factory=dict,
        description="Border mapping for specific colors"
    )


class TileConfig(BaseModel):
    types: int = Field(36)
    cat_types: int = Field(6)
    identical_tiles: int = Field(2)
    identical_cat_tiles: int = Field(1)
    patterns: int = Field(3)
    colors: int = Field(3)
    none_left: int = Field(0)


class EvaluationConfig(BaseModel):
    weight_final_score: float = 1.0
    weight_cat_potential: float = 1.0
    weight_color_potential: float = 0.1
    weight_objective_viability: float = 1.0


class GameSettings(BaseModel):
    name:str = Field("")
    player_hand_size: int = 2
    nr_of_tiles_in_shop: int = 3
    max_players: int = 4
    min_region_for_scoring: int = 3

    board: BoardConfig
    tiles: TileConfig
    evaluation: EvaluationConfig