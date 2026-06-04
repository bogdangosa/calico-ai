import random
import pickle
import numpy as np
from typing import Dict, Optional, Any, List, Tuple

from src.agents.agent_base import AgentBase
from src.engine.environments.canonical_transformer import CanonicalTransformer
from src.models.game_models import CalicoAction, ActionType

class TabularQLearningAgent(AgentBase):
    """
    A Q-Learning agent that uses a state-action (Q-table) to store values.
    Indexed as: ((state_key), (action_key))
    """
    def __init__(
        self, 
        config, 
        learning_rate: float = 0.1, 
        discount_factor: float = 0.95, 
        epsilon: float = 0.1,
        q_table_path: Optional[str] = None
    ):
        super().__init__(config)
        self.lr = learning_rate
        self.gamma = discount_factor
        self.epsilon = epsilon
        self.transformer = CanonicalTransformer(config)
        self.q_table: Dict[Tuple[Tuple, Tuple], float] = {}
        
        if q_table_path:
            self.load(q_table_path)

    def _get_state_key(self, env) -> tuple:
        """Extracts and returns a flat 4-element tuple of the inner board positions."""
        inner_board = self.transformer.get_inner_board(env.board_matrix)
        return tuple(inner_board.flatten())

    def get_action_key(self, action: CalicoAction, env) -> tuple:
        """Extracts action mechanics into a hashable 4-element tuple."""
        # Resolve tile_id from tile_index
        tile_id = -1
        if action.action_type == ActionType.PLACE:
            tile_id = env.player_tiles[action.tile_index]
        elif action.action_type == ActionType.BUY:
            tile_id = env.shop_tiles[action.tile_index]
            
        return (action.action_type.value, tile_id, action.row if action.row is not None else -1, action.col if action.col is not None else -1)

    def get_q_value(self, state_key: tuple, action_key: tuple) -> float:
        """Returns the stored value for a state-action pair, or 0.0 if unknown."""
        return self.q_table.get((state_key, action_key), 0.0)

    def _get_best_action(self, env, legal_actions: List[CalicoAction], state_key: tuple) -> CalicoAction:
        """Locates the maximum Q-value entry with random tie-breaking logic."""
        best_q = -float('inf')
        best_actions = []

        for action in legal_actions:
            action_key = self.get_action_key(action, env)
            q_val = self.get_q_value(state_key, action_key)
            
            if q_val > best_q:
                best_q = q_val
                best_actions = [action]
            elif q_val == best_q:
                best_actions.append(action)
                
        return random.choice(best_actions)

    def select_action(self, env) -> Optional[CalicoAction]:
        """Implements epsilon-greedy exploration."""
        legal_actions = env.get_legal_actions()
        if not legal_actions:
            return None
            
        if random.random() < self.epsilon:
            return random.choice(legal_actions)
            
        state_key = self._get_state_key(env)
        return self._get_best_action(env, legal_actions, state_key)

    def update(
        self, 
        state_key: tuple, 
        action_key: tuple, 
        reward: float, 
        next_env,
        next_state_key: tuple, 
        next_legal_actions: List[CalicoAction], 
        done: bool
    ):
        """Standard TD Q-Learning update rule."""
        current_q = self.get_q_value(state_key, action_key)
        
        if done or not next_legal_actions:
            target = reward
        else:
            # max Q(s', a')
            max_next_q = max(
                self.get_q_value(next_state_key, self.get_action_key(a, next_env)) 
                for a in next_legal_actions
            )
            target = reward + self.gamma * max_next_q
            
        # Q(s,a) = Q(s,a) + alpha * (target - Q(s,a))
        self.q_table[(state_key, action_key)] = current_q + self.lr * (target - current_q)

    def save(self, path: str):
        """Saves the Q-table to a file using pickle."""
        with open(path, 'wb') as f:
            pickle.dump(self.q_table, f)

    def load(self, path: str):
        """Loads the Q-table from a file."""
        try:
            with open(path, 'rb') as f:
                self.q_table = pickle.load(f)
        except (FileNotFoundError, EOFError):
            print(f"Warning: Could not load Q-table from {path}. Starting fresh.")
            self.q_table = {}
