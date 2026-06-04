from enum import Enum

from pydantic import BaseModel
from typing import Optional

class ActionType(str, Enum):
    PLACE = "PLACE"
    BUY = "BUY"

class CalicoAction(BaseModel):
    action_type: ActionType
    tile_index: int
    row: Optional[int] = None
    col: Optional[int] = None
