import uuid
from typing import List, Dict, Any, Optional

from src.agents.agent_base import AgentBase
from src.engine.environments.calico_env import CalicoEnv
from src.models.game_config import GameSettings
from src.models.game_models import CalicoAction

class GameInstance:
    def __init__(
        self, 
        game_id: uuid.UUID, 
        config_type: str, 
        settings: GameSettings, 
        nr_of_players: int, 
        bot_types: List[str]
    ):
        self.game_id = game_id
        self.config_type = config_type
        self.settings = settings
        self.nr_of_players = nr_of_players
        self.bot_types = bot_types
        self.env = CalicoEnv(settings)

        self.env.start_game()

        self.player_agents_list = [] 

    def get_state(self) -> Dict[str, Any]:
        """Returns the current state of the game instance."""
        return {
            "game_id": str(self.game_id),
            "config_type": self.config_type,
            "mode": self.env.mode,
            "player_tiles": self.env.player_tiles,
            "shop_tiles": self.env.shop_tiles,
            "board": self.env.board_matrix.tolist() if hasattr(self.env.board_matrix, 'tolist') else self.env.board_matrix,
            "is_game_over": self.env.is_game_over()
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
