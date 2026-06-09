from pydantic import BaseModel, Field
from datetime import datetime
from typing import List, Optional, Dict, Any
from uuid import UUID
from src.models.game_models import ActionType

class StartGameRequest(BaseModel):
    nr_of_players: int = Field(..., ge=1, le=4)
    bot_types: List[str] = Field(default_factory=list)
    configuration_type: str = Field(..., pattern="^(mini|micro|full)$")

class StartGameResponse(BaseModel):
    game_id: UUID
    game_code: str

class PerformActionRequest(BaseModel):
    action_type: ActionType
    tile_index: int
    row: Optional[int] = None
    col: Optional[int] = None

class GameStateResponse(BaseModel):
    game_id: str
    game_code: str
    config_type: str
    mode: str
    player_tiles: List[int]
    cat_tiles: List[int]
    shop_tiles: List[int]
    board: List[List[int]]
    is_game_over: bool
    score: Any

class GameSummary(BaseModel):
    game_id: UUID
    game_code: str
    config_type: str
    nr_of_players: int
    is_game_over: bool
    created_at: datetime

class GamesListResponse(BaseModel):
    games: List[GameSummary]

# Player Models
class PlayerBase(BaseModel):
    player_name: str
    player_type: str = Field(..., pattern="^(ai|person)$")
    order_index: Optional[int] = None

class PlayerCreate(PlayerBase):
    pass

class PlayerResponse(PlayerBase):
    id: UUID
    game_id: UUID

    class Config:
        from_attributes = True
