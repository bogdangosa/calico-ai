import uuid
import numpy as np
from typing import List, Dict, Any, Optional

from src.agents.agent_base import AgentBase
from src.engine.environments.calico_env import CalicoEnv
from src.engine.scoring.scoring import ScoringCalculator
from src.models.game_config import GameSettings
from src.models.game_models import CalicoAction

class GameInstance:
    def __init__(
        self, 
        game_id: uuid.UUID, 
        game_code: str,
        config_type: str, 
        settings: GameSettings, 
        nr_of_players: int, 
        bot_types: List[str]
    ):
        self.scoring_calculator = ScoringCalculator(settings)
        self.game_id = game_id
        self.game_code = game_code
        self.config_type = config_type
        self.settings = settings
        self.nr_of_players = nr_of_players
        self.bot_types = bot_types
        self.env = CalicoEnv(settings)
        self.player_agents_list = [] 

    def start_new_game(self):
        self.env.start_game()

    def get_full_state(self) -> Dict[str, Any]:
        """Returns the complete state of the game for persistence."""
        return {
            "game_id": str(self.game_id),
            "game_code": self.game_code,
            "config_type": self.config_type,
            "nr_of_players": self.nr_of_players,
            "bot_types": self.bot_types,
            "env_state": {
                "mode": self.env.mode,
                "player_tiles": self.env.player_tiles,
                "shop_tiles": self.env.shop_tiles.tolist() if isinstance(self.env.shop_tiles, np.ndarray) else self.env.shop_tiles,
                "tile_pool": self.env.tile_pool.tolist() if isinstance(self.env.tile_pool, np.ndarray) else self.env.tile_pool,
                "cat_tiles": self.env.cat_tiles.tolist() if isinstance(self.env.cat_tiles, np.ndarray) else self.env.cat_tiles,
                "board_matrix": self.env.board_matrix.tolist() if isinstance(self.env.board_matrix, np.ndarray) else self.env.board_matrix,
                "selected_player_tile_index": self.env.selected_player_tile_index,
                "is_game_over": self.env.is_game_over()
            }
        }

    @classmethod
    def from_state(cls, state: Dict[str, Any], settings: GameSettings) -> 'GameInstance':
        """Reconstructs a GameInstance from a persisted state."""
        instance = cls(
            game_id=uuid.UUID(state["game_id"]),
            game_code=state.get("game_code", ""), # Fallback for old states
            config_type=state["config_type"],
            settings=settings,
            nr_of_players=state["nr_of_players"],
            bot_types=state["bot_types"]
        )
        
        env_state = state["env_state"]
        instance.env.mode = env_state["mode"]
        instance.env.player_tiles = env_state["player_tiles"]
        instance.env.shop_tiles = np.array(env_state["shop_tiles"])
        instance.env.tile_pool = np.array(env_state["tile_pool"])
        instance.env.cat_tiles = np.array(env_state["cat_tiles"])
        instance.env.board_matrix = np.array(env_state["board_matrix"])
        instance.env.selected_player_tile_index = env_state["selected_player_tile_index"]
        
        return instance

    def get_state(self) -> Dict[str, Any]:
        """Returns the current state of the game instance for API responses."""
        return {
            "game_id": str(self.game_id),
            "game_code": self.game_code,
            "config_type": self.config_type,
            "mode": self.env.mode,
            "player_tiles": self.env.player_tiles,
            "cat_tiles": self.env.cat_tiles,
            "shop_tiles": self.env.shop_tiles.tolist() if isinstance(self.env.shop_tiles, np.ndarray) else self.env.shop_tiles,
            "board": self.env.board_matrix.tolist() if isinstance(self.env.board_matrix, np.ndarray) else self.env.board_matrix,
            "is_game_over": self.env.is_game_over(),
            "score": self.scoring_calculator.get_total_detailed_score(self.env.board_matrix, self.env.cat_tiles)[0],
        }

    def perform_action(self, action: CalicoAction):
        """Executes a manual action in the environment."""
        self.env.perform_action(action)

    def perform_ai_action(self, agent: AgentBase):
        """Executes an action using the provided agent."""
        action = agent.select_action(self.env)
        if action:
            self.env.perform_action(action)
        return action
