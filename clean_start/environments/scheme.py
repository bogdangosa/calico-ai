from typing import List

from pydantic import BaseModel


class CalicoAction(BaseModel):
    action_type: str
    tile_index: int
    row: int
    col: int

class CalicoEnv(BaseModel):
    tile_pool: str
    player_tiles: List[str]
    board_matrix: List[str]
    move_history: List[CalicoAction]

