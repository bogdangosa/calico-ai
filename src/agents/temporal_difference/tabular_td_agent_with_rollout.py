import random
import pickle
import numpy as np
from typing import Dict, Optional, Any

from src.agents.agent_base import AgentBase
from src.engine.environments.canonical_transformer import CanonicalTransformer
from src.models.game_models import CalicoAction
from src.engine.scoring.scoring import ScoringCalculator


class TabularTDAgentWithRollout(AgentBase):

    def __init__(
            self,
            config,
            learning_rate: float = 0.1,
            discount_factor: float = 0.95,
            epsilon: float = 0.1,
            v_table_path: Optional[str] = None
    ):
        super().__init__(config)
        self.lr = learning_rate
        self.gamma = discount_factor
        self.epsilon = epsilon
        self.transformer = CanonicalTransformer(config)
        self.scorer = ScoringCalculator(config)
        self.v_table: Dict[tuple, float] = {}

        if v_table_path:
            self.load(v_table_path)

    def _get_state_key(self, env) -> tuple:
        playable_board = self.transformer.get_inner_board(env.board_matrix)
        flat_board = playable_board.flatten()
        sorted_hand = sorted(env.player_tiles)
        return tuple(flat_board) + tuple(sorted_hand)

    def get_v_value(self, state_key: tuple) -> float:
        return self.v_table.get(state_key, 0.0)

    def get_state_value(self, env, state_key: tuple) -> float:
        if env.is_game_over():
            return self.scorer.evaluate_move(env.board_matrix, env.cat_tiles)
        return self.get_v_value(state_key)

    def select_action(self, env) -> Optional[CalicoAction]:
        legal_actions = env.get_legal_actions()
        if not legal_actions:
            return None

        if random.random() < self.epsilon:
            return random.choice(legal_actions)

        return self._get_best_action(env, legal_actions)

    def _get_best_action(self, env, legal_actions: list[CalicoAction]) -> CalicoAction:
        best_value = -float('inf')
        best_actions = []
        history_was_enabled = env.history_manager is not None

        if not history_was_enabled:
            env.enable_history()

        for action in legal_actions:
            env.perform_action(action)
            state_key = self._get_state_key(env)
            value = self.get_state_value(env, state_key)
            env.undo_action()

            if value > best_value:
                best_value = value
                best_actions = [action]
            elif value == best_value:
                best_actions.append(action)

        if not history_was_enabled:
            env.disable_history()

        return random.choice(best_actions)

    def update(self, state_key: tuple, reward: float, next_board_matrix: np.ndarray, done: bool):
        current_v = self.get_v_value(state_key)

        if done:
            target = reward
        else:
            playable_board = self.transformer.get_inner_board(next_board_matrix)
            flat_board = playable_board.flatten()
            sample_values = []

            for _ in range(10):
                random_hand = sorted(random.choices(range(self.config.tiles.types), k=2))
                sample_key = tuple(flat_board) + tuple(random_hand)
                sample_values.append(self.get_v_value(sample_key))

            target = self.gamma * np.mean(sample_values)

        new_v = current_v + self.lr * (target - current_v)
        self.v_table[state_key] = new_v

    def save(self, path: str):
        with open(path, 'wb') as f:
            pickle.dump(self.v_table, f)

    def load(self, path: str):
        try:
            with open(path, 'rb') as f:
                self.v_table = pickle.load(f)
        except (FileNotFoundError, EOFError):
            print(f"Warning: Could not load V-table from {path}. Starting fresh.")
            self.v_table = {}