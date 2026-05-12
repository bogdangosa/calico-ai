import uuid
from typing import Dict, List, Optional
from src.services.game_instance import GameInstance
from src.utils.config import load_config
from src.engine.scoring.scoring import ScoringCalculator
from src.agents.random_agent import RandomAgent
from src.agents.lookahead.one_step_lookahead_agent import OneStepLookaheadAgent
from src.models.game_models import CalicoAction
from src.api.models import GameSummary

class GameService:
    def __init__(self):
        self.games: Dict[uuid.UUID, GameInstance] = {}
        self.config_paths = {
            "mini": "config/mini_calico_settings.json",
            "micro": "config/micro_calico_settings.json",
            "full": "config/calico_settings.json"
        }

    def start_game(self, nr_of_players: int, bot_types: List[str], configuration_type: str) -> uuid.UUID:
        game_id = uuid.uuid4()
        config_path = self.config_paths.get(configuration_type.lower())
        if not config_path:
            raise ValueError(f"Invalid configuration type: {configuration_type}")
        
        settings = load_config(config_path)
        game_instance = GameInstance(game_id, configuration_type, settings, nr_of_players, bot_types)

        self._initialize_game_agents(game_instance)
        
        self.games[game_id] = game_instance
        return game_id

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

    def get_all_games(self) -> List[GameSummary]:
        return [
            GameSummary(
                game_id=g.game_id,
                config_type=g.config_type,
                nr_of_players=g.nr_of_players,
                is_game_over=g.env.is_game_over()
            )
            for g in self.games.values()
        ]

    def get_game_state(self, game_id: uuid.UUID) -> Dict:
        game = self._get_game_or_raise(game_id)
        return game.get_state()

    def perform_action(self, game_id: uuid.UUID, action: CalicoAction):
        game = self._get_game_or_raise(game_id)
        game.perform_action(action)

    def perform_ai_agent_action(self, game_id: uuid.UUID):
        game = self._get_game_or_raise(game_id)

        if hasattr(game, 'player_agents_list') and game.player_agents_list:
            agent = game.player_agents_list[0]
            return game.perform_ai_action(agent)
        else:
            raise ValueError("No AI agents configured for this game.")

    def _get_game_or_raise(self, game_id: uuid.UUID) -> GameInstance:
        if game_id not in self.games:
            raise KeyError(f"Game with ID {game_id} not found.")
        return self.games[game_id]
