from fastapi import APIRouter, HTTPException, Depends
from uuid import UUID
from typing import List
from sqlalchemy.ext.asyncio import AsyncSession

from src.api.models import (
    StartGameRequest, 
    StartGameResponse, 
    GameStateResponse, 
    PerformActionRequest,
    GamesListResponse,
    PlayerCreate,
    PlayerResponse
)
from src.services.game_service import GameService
from src.models.game_models import CalicoAction
from src.repository.database.connection import get_session
from src.api.sockets import manager

router = APIRouter()

async def get_game_service(session: AsyncSession = Depends(get_session)) -> GameService:
    return GameService(session)

@router.post("/start-game", response_model=StartGameResponse)
async def start_game(
    request: StartGameRequest, 
    service: GameService = Depends(get_game_service)
):
    try:
        response = await service.start_game(
            nr_of_players=request.nr_of_players,
            bot_types=request.bot_types,
            configuration_type=request.configuration_type
        )
        return response
    except ValueError as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/get-current-games", response_model=GamesListResponse)
async def get_current_games(service: GameService = Depends(get_game_service)):
    games = await service.get_all_games()
    return GamesListResponse(games=games)

@router.delete("/delete-all-games")
async def delete_all_games(service: GameService = Depends(get_game_service)):
    try:
        await service.delete_all_games()
        return {"status": "success", "message": "All games deleted."}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.delete("/delete-game/{game_code}")
async def delete_game(game_code: str, service: GameService = Depends(get_game_service)):
    try:
        await service.delete_game_by_code(game_code)
        return {"status": "success", "message": f"Game {game_code} deleted."}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/get-game-state/{game_code}", response_model=GameStateResponse)
async def get_game_state(
    game_code: str, 
    service: GameService = Depends(get_game_service)
):
    try:
        state = await service.get_game_state(game_code)
        return GameStateResponse(**state)
    except KeyError as e:
        raise HTTPException(status_code=404, detail=str(e))

@router.post("/game/{game_code}/add-player", response_model=PlayerResponse)
async def add_player(
    game_code: str,
    request: PlayerCreate,
    service: GameService = Depends(get_game_service)
):
    try:
        player = await service.add_player_to_game(game_code, request)
        await manager.broadcast(game_code, {
            "event": "player_joined",
            "player_name": player.player_name,
            "player_type": player.player_type
        })
        return player
    except KeyError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/perform-action/{game_code}")
async def perform_action(
    game_code: str, 
    request: PerformActionRequest, 
    service: GameService = Depends(get_game_service)
):
    try:
        action = CalicoAction(
            action_type=request.action_type,
            tile_index=request.tile_index,
            row=request.row,
            col=request.col
        )
        await service.perform_action(game_code, action)
        await manager.broadcast(game_code, {"event": "state_updated"})
        return {"status": "success"}
    except KeyError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/perform-ai-agent-action/{game_code}")
async def perform_ai_agent_action(
    game_code: str, 
    service: GameService = Depends(get_game_service)
):
    try:
        action = await service.perform_ai_agent_action(game_code)
        await manager.broadcast(game_code, {"event": "state_updated"})
        return {"status": "success", "action": action.dict() if action else None}
    except KeyError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except ValueError as e:
        print(e)
        raise HTTPException(status_code=500, detail=str(e))
    except Exception as e:
        print(e)
        raise HTTPException(status_code=500, detail=str(e))
