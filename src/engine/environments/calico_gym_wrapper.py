import gymnasium as gym
import numpy as np
from gymnasium import spaces
from typing import Optional, Dict, Any, Tuple

from src.engine.environments.calico_env import CalicoEnv
from src.engine.environments.action_mapper import ActionMapper
from src.engine.environments.calico_encoder import CalicoEncoder
from src.engine.scoring.potential_scoring import PotentialScoringCalculator
from src.engine.scoring.scoring import ScoringCalculator

from src.models.game_models import CalicoAction, ActionType

class CalicoGymWrapper(gym.Env):
    """
    Gymnasium wrapper for CalicoEnv to make it compatible with Stable-Baselines3.
    """
    metadata = {"render_modes": ["human"]}

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.env = CalicoEnv(config)
        self.reward_scorer = PotentialScoringCalculator(config)
        self.actual_scorer = ScoringCalculator(config)
        
        # Initialize mapper and encoder
        self.mapper = ActionMapper(
            board_size=config.board.size,
            hand_size=config.player_hand_size,
            shop_size=config.nr_of_tiles_in_shop
        )
        self.encoder = CalicoEncoder(config)
        
        # Action space: Dynamic based on config
        self.action_space = spaces.Discrete(self.mapper.total_actions)
        
        # Start game briefly to get flat features size
        self.env.start_game()
        
        mode_size = 1
        hand_size = len(self.env.player_tiles)
        shop_size = len(self.env.shop_tiles)
        
        if isinstance(self.env.cat_tiles, np.ndarray):
            cat_size = self.env.cat_tiles.size
        else:
            cat_size = len(self.env.cat_tiles)
            
        flat_features_size = mode_size + hand_size + shop_size + cat_size
        
        self.observation_space = spaces.Dict({
            "board": spaces.Box(
                low=0, 
                high=1, 
                shape=(config.board.size, config.board.size, self.encoder.total_feature_layers), 
                dtype=np.float32
            ),
            "flat_features": spaces.Box(
                low=-100, 
                high=100, 
                shape=(flat_features_size,), 
                dtype=np.int32
            ),
            "action_mask": spaces.Box(
                low=0, 
                high=1, 
                shape=(self.mapper.total_actions,), 
                dtype=np.int8
            )
        })

        self.last_heuristic_score = 0.0

    def get_flat_features(self) -> np.ndarray:
        mode = np.array([int(self.env.mode == ActionType.BUY)], dtype=np.int32)
        player_tiles = np.array(self.env.player_tiles, dtype=np.int32)
        shop_tiles = np.array(self.env.shop_tiles, dtype=np.int32)
        
        if isinstance(self.env.cat_tiles, np.ndarray):
            cat_tiles = self.env.cat_tiles.flatten().astype(np.int32)
        else:
            cat_tiles = np.array(self.env.cat_tiles, dtype=np.int32).flatten()
            
        return np.concatenate([mode, player_tiles, shop_tiles, cat_tiles])

    def action_masks(self) -> np.ndarray:
        """Converts legal actions into a boolean mask."""
        mask = np.zeros(self.mapper.total_actions, dtype=np.int8)
        legal_actions = self.env.get_legal_actions()
        
        for action in legal_actions:
            idx = self._map_action_to_index(action)
            if idx is not None:
                mask[idx] = 1
        return mask

    def _map_action_to_index(self, action: CalicoAction) -> Optional[int]:
        return self.mapper.action_to_index(action)

    def _map_index_to_action(self, action_idx: int) -> Optional[CalicoAction]:
        return self.mapper.index_to_action(action_idx)

    def reset(self, seed: Optional[int] = None, options: Optional[Dict[str, Any]] = None) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """Resets the environment and returns the initial observation and info."""
        super().reset(seed=seed)
        if seed is not None:
            np.random.seed(seed)
            
        self.env.start_game(seed=seed if seed is not None else 41)

        self.last_heuristic_score = self.reward_scorer.evaluate_move(
            self.env.board_matrix, self.env.cat_tiles
        )
        
        observation = {
            "board": self.encoder.encode(self.env),
            "flat_features": self.get_flat_features(),
            "action_mask": self.action_masks()
        }
        return observation, {}

    def step(self, action_idx: int) -> Tuple[Dict[str, Any], float, bool, bool, Dict[str, Any]]:
        """Performs an action and returns (obs, reward, terminated, truncated, info)."""
        action = self._map_index_to_action(action_idx)
        
        # Execute action
        if action:
            self.env.perform_action(action)
        
        # Calculate reward
        current_heuristic = self.reward_scorer.evaluate_move(
            self.env.board_matrix, self.env.cat_tiles
        )
        reward = float(current_heuristic - self.last_heuristic_score)
        self.last_heuristic_score = current_heuristic
        
        terminated = self.env.is_game_over()
        truncated = False
            
        observation = {
            "board": self.encoder.encode(self.env),
            "flat_features": self.get_flat_features(),
            "action_mask": self.action_masks()
        }

        actual_score = self.actual_scorer.evaluate_move(
            self.env.board_matrix, self.env.cat_tiles
        )
        info = {"score": actual_score}
        
        return observation, reward, terminated, truncated, info

    def render(self):
        """Simple text-based render."""
        print(self.env)
