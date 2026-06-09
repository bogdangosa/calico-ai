import random
import pickle
import numpy as np
from typing import Dict, Optional, Any, List, Tuple

from src.agents.q_learning.tabular_q_learning_agent import TabularQLearningAgent
from src.engine.environments.canonical_transformer import CanonicalTransformer
from src.models.game_models import CalicoAction, ActionType
from src.engine.scoring.scoring import ScoringCalculator

class TabularQLearningAgentTerminalAnchoring(TabularQLearningAgent):
    """
    A Tabular Q-Learning agent with Terminal Anchoring.
    """
    def __init__(
        self, 
        config, 
        learning_rate: float = 0.1, 
        discount_factor: float = 0.95, 
        epsilon: float = 0.1,
        q_table_path: Optional[str] = None
    ):
        super().__init__(config, learning_rate, discount_factor, epsilon, q_table_path)
        self.scorer = ScoringCalculator(config)

    def _get_state_key(self, env) -> tuple:
        """Includes hand in the state key for better state differentiation."""
        inner_board = self.transformer.get_inner_board(env.board_matrix)
        hand = tuple(sorted(env.player_tiles))
        return tuple(inner_board.flatten()) + hand

    def get_grounded_q_value(self, env, state_key: tuple, action: CalicoAction) -> float:
        """
        If the action ends the game, returns the ACTUAL final score.
        Otherwise, returns the stored Q-value.
        """
        env.enable_history()
        env.perform_action(action)
        is_done = env.is_game_over()
        
        if is_done:
            score, *_ = self.scorer.get_total_detailed_score(env.board_matrix, env.cat_tiles)
            env.undo_action()
            return float(score)
        
        env.undo_action()
        action_key = self.get_action_key(action, env)
        return self.get_q_value(state_key, action_key)

    def _get_best_action(self, env, legal_actions: List[CalicoAction], state_key: tuple) -> CalicoAction:
        """Uses grounded evaluation for terminal moves."""
        best_q = -float('inf')
        best_actions = []

        for action in legal_actions:
            q_val = self.get_grounded_q_value(env, state_key, action)
            
            if q_val > best_q:
                best_q = q_val
                best_actions = [action]
            elif q_val == best_q:
                best_actions.append(action)
                
        return random.choice(best_actions)
