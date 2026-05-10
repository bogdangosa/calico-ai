from fastapi import APIRouter, HTTPException, Depends
from uuid import UUID
from typing import List

from src.api.models import (
    StartGameRequest, 
    StartGameResponse, 
    GameStateResponse, 
    PerformActionRequest
)
from src.services.game_service import GameService
from src.models.game_models import CalicoAction

router = APIRouter()

game_service = GameService()

@router.post("/start-game", response_model=StartGameResponse)
async def start_game(request: StartGameRequest):
    try:
        game_id = game_service.start_game(
            nr_of_players=request.nr_of_players,
            bot_types=request.bot_types,
            configuration_type=request.configuration_type
        )
        return StartGameResponse(game_id=game_id)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

@router.get("/get-game-state/{game_id}", response_model=GameStateResponse)
async def get_game_state(game_id: UUID):
    try:
        state = game_service.get_game_state(game_id)
        return GameStateResponse(**state)
    except KeyError as e:
        raise HTTPException(status_code=404, detail=str(e))

@router.post("/perform-action/{game_id}")
async def perform_action(game_id: UUID, request: PerformActionRequest):
    try:
        action = CalicoAction(
            action_type=request.action_type,
            tile_index=request.tile_index,
            row=request.row,
            col=request.col
        )
        game_service.perform_action(game_id, action)
        return {"status": "success"}
    except KeyError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@router.post("/perform-ai-agent-action/{game_id}")
async def perform_ai_agent_action(game_id: UUID):
    try:
        action = game_service.perform_ai_agent_action(game_id)
        return {"status": "success", "action": action.dict() if action else None}
    except KeyError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
