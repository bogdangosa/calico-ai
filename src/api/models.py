from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
from uuid import UUID
from src.models.game_models import ActionType

class StartGameRequest(BaseModel):
    nr_of_players: int = Field(..., ge=1, le=4)
    bot_types: List[str] = Field(default_factory=list)
    configuration_type: str = Field(..., pattern="^(mini|micro|full)$")

class StartGameResponse(BaseModel):
    game_id: UUID

class PerformActionRequest(BaseModel):
    action_type: ActionType
    tile_index: int
    row: Optional[int] = None
    col: Optional[int] = None

class GameStateResponse(BaseModel):
    game_id: str
    config_type: str
    mode: str
    player_tiles: List[int]
    shop_tiles: List[int]
    board: List[List[int]]
    is_game_over: bool
