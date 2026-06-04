import uuid
import random
import string
from typing import Dict, List, Optional
from sqlalchemy.ext.asyncio import AsyncSession

from src.agents.agent_factory import AgentFactory
from src.services.game_instance import GameInstance
from src.utils.config import load_config
from src.engine.scoring.scoring import ScoringCalculator
from src.agents.random_agent import RandomAgent
from src.agents.lookahead.one_step_lookahead_agent import OneStepLookaheadAgent
from src.models.game_models import CalicoAction
from src.api.models import GameSummary, StartGameResponse, PlayerCreate, PlayerResponse
from src.repository.game_repository import GameRepository
from src.repository.player_repository import PlayerRepository
from src.utils.sanitize import sanitize_for_json


class GameService:
    def __init__(self, session: AsyncSession):
        self.repository = GameRepository(session)
        self.player_repository = PlayerRepository(session)
        self.config_paths = {
            "mini": "config/mini_calico_settings.json",
            "micro": "config/micro_calico_settings.json",
            "full": "config/calico_settings.json"
        }

    async def _generate_unique_code(self) -> str:
        """Generates a unique 4-character alphanumeric code."""
        chars = string.ascii_uppercase + string.digits
        while True:
            code = ''.join(random.choices(chars, k=4))
            if not await self.repository.is_code_taken(code):
                return code

    async def start_game(self, nr_of_players: int, bot_types: List[str], configuration_type: str) -> StartGameResponse:
        config_path = self.config_paths.get(configuration_type.lower())
        if not config_path:
            raise ValueError(f"Invalid configuration type: {configuration_type}")
        
        settings = load_config(config_path)
        game_id = uuid.uuid4()
        game_code = await self._generate_unique_code()

        
        game_instance = GameInstance(game_id, game_code, configuration_type, settings, nr_of_players, bot_types)
        game_instance.start_new_game()
        
        state = game_instance.get_full_state()
        await self.repository.create(
            game_code=game_code,
            config_type=configuration_type,
            nr_of_players=nr_of_players,
            bot_types=bot_types,
            state=state
        )
        return StartGameResponse(game_id=game_id, game_code=game_code)

    async def add_player_to_game(self, game_code: str, player_data: PlayerCreate) -> PlayerResponse:
        """Adds a player to an existing game using game_code."""
        game_orm = await self.repository.get_by_code(game_code)
        if not game_orm:
            raise KeyError(f"Game with code {game_code} not found.")
        
        if player_data.order_index is None:
            existing_players = await self.player_repository.get_players_by_game(game_orm.id)
            if not existing_players:
                player_data.order_index = 0
            else:
                max_index = max(p.order_index for p in existing_players)
                player_data.order_index = max_index + 1
        
        player_orm = await self.player_repository.add_player(game_orm.id, player_data)
        return PlayerResponse.model_validate(player_orm)

    async def get_all_games(self) -> List[GameSummary]:
        game_orms = await self.repository.list_all()
        return [
            GameSummary(
                game_id=g.id,
                game_code=g.game_code,
                config_type=g.config_type,
                nr_of_players=g.nr_of_players,
                is_game_over=g.state.get("env_state", {}).get("is_game_over", False)
            )
            for g in game_orms
        ]

    async def delete_all_games(self) -> None:
        await self.repository.delete_all()

    async def delete_game_by_code(self, game_code: str) -> None:
        await self.repository.delete_by_code(game_code)

    async def get_game_state(self, game_code: str) -> Dict:
        game_instance = await self._load_game_by_code(game_code)
        return game_instance.get_state()

    async def perform_action(self, game_code: str, action: CalicoAction):
        game_orm = await self.repository.get_by_code(game_code)
        if not game_orm:
            raise KeyError(f"Game with code {game_code} not found.")

        settings = load_config(self.config_paths.get(game_orm.config_type.lower()))
        game_instance = GameInstance.from_state(game_orm.state, settings)

        game_instance.perform_action(action)

        raw_state = game_instance.get_full_state()

        safe_state = sanitize_for_json(raw_state)

        await self.repository.update_state(game_orm.id, safe_state)


    async def perform_ai_agent_action_for_player(self, game_code: str,agent_type: str):
        game_orm = await self.repository.get_by_code(game_code)
        if not game_orm:
            raise KeyError(f"Game with code {game_code} not found.")

        settings = load_config(self.config_paths.get(game_orm.config_type.lower()))
        game_instance = GameInstance.from_state(game_orm.state, settings)

        ai_agent = AgentFactory.create_agent(agent_type=agent_type,game_config=settings)
        action = ai_agent.select_action(game_instance.env)
        print(action)
        await self.perform_action(game_code, action)


    async def perform_ai_agent_action(self, game_code: str):
        game_orm = await self.repository.get_by_code(game_code)
        if not game_orm:
            raise KeyError(f"Game with code {game_code} not found.")
            
        settings = load_config(self.config_paths.get(game_orm.config_type.lower()))
        game_instance = GameInstance.from_state(game_orm.state, settings)
        self._initialize_game_agents(game_instance)

        if hasattr(game_instance, 'player_agents_list') and game_instance.player_agents_list:
            agent = game_instance.player_agents_list[0]
            action = game_instance.perform_ai_action(agent)
            await self.repository.update_state(game_orm.id, game_instance.get_full_state())
            return action
        else:
            raise ValueError("No AI agents configured for this game.")

    async def _load_game_by_code(self, game_code: str) -> GameInstance:
        game_orm = await self.repository.get_by_code(game_code)
        if not game_orm:
            raise KeyError(f"Game with code {game_code} not found.")
        
        config_path = self.config_paths.get(game_orm.config_type.lower())
        settings = load_config(config_path)
        
        return GameInstance.from_state(game_orm.state, settings)

    def _initialize_game_agents(self, game: GameInstance):
        """Initializes agents based on bot_types."""
        agents = []
        scorer = ScoringCalculator(game.settings)
        
        for bot_type in game.bot_types:
            if bot_type == "random":
                agents.append(RandomAgent(game.settings))
            elif bot_type == "one_step_lookahead":
                agents.append(OneStepLookaheadAgent(scorer, game.settings))
            else:
                agents.append(RandomAgent(game.settings))
        
        game.player_agents_list = agents
