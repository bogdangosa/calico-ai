import torch
import numpy as np
import random
from typing import List, Optional
from src.agents.agent_base import AgentBase
from src.machine_learning.networks.baseline_q_network import BaselineQNetwork
from src.engine.environments.calico_encoder import CalicoEncoder
from src.models.game_models import CalicoAction, ActionType

class BaselineQLearningAgent(AgentBase):

    def __init__(self, config, model_path=None, epsilon=0.1, device="cpu"):
        super().__init__(config)
        self.encoder = CalicoEncoder(config)
        self.device = torch.device(device)

        inner_size = config.board.size - 2
        self.action_space_size = config.player_hand_size * (inner_size ** 2) + config.nr_of_tiles_in_shop

        self.model = BaselineQNetwork(
            input_channels=self.encoder.total_feature_layers,
            board_size=config.board.size,
            flat_features_size=self.encoder.flat_features_size,
            action_space_size=self.action_space_size
        ).to(self.device)

        if model_path:
            self.model.load(model_path)

        self.epsilon = epsilon
        self.model.eval()

    def _get_action_index(self, action: CalicoAction) -> int:
        inner_size = self.config.board.size - 2
        num_place_actions = self.config.player_hand_size * (inner_size ** 2)

        if action.action_type == ActionType.BUY:
            return num_place_actions + action.tile_index

        inner_row = action.row - 1
        inner_col = action.col - 1
        slot_index = inner_row * inner_size + inner_col

        return action.tile_index * (inner_size ** 2) + slot_index

    def get_action_mask(self, env) -> torch.Tensor:
        inner_size = self.config.board.size - 2
        num_place_actions = self.config.player_hand_size * (inner_size ** 2)
        total_actions = num_place_actions + self.config.nr_of_tiles_in_shop

        mask = torch.zeros(total_actions, dtype=torch.bool)
        legal_actions = env.get_legal_actions()
        for action in legal_actions:
            idx = self._get_action_index(action)
            mask[idx] = True
        return mask

    def select_action(self, env) -> Optional[CalicoAction]:
        legal_actions = env.get_legal_actions()
        if not legal_actions:
            return None

        if random.random() < self.epsilon:
            return random.choice(legal_actions)

        return self._get_best_action(env, legal_actions)

    def _get_best_action(self, env, legal_actions: List[CalicoAction]) -> CalicoAction:
        board_tensor, flat_tensor = self.get_state_tensors(env)

        with torch.no_grad():
            q_values = self.model(board_tensor, flat_tensor).squeeze(0)

        mask = torch.full_like(q_values, float('-inf'))
        action_mapping = {}

        for action in legal_actions:
            idx = self._get_action_index(action)
            mask[idx] = 0.0
            action_mapping[idx] = action

        masked_q_values = q_values + mask
        
        # Check for NaNs or all -inf to prevent crash
        if torch.isnan(masked_q_values).any() or torch.all(torch.isinf(masked_q_values)):
            return random.choice(legal_actions)

        best_idx = torch.argmax(masked_q_values).item()
        
        # Fallback if argmax still returns something invalid
        return action_mapping.get(best_idx, random.choice(legal_actions))

    def get_state_tensors(self, env):
        # Board tensor
        encoded = self.encoder.encode(env)
        encoded = np.transpose(encoded, (2, 0, 1))
        board_tensor = torch.from_numpy(encoded).unsqueeze(0).to(self.device).float()

        # Flat features tensor
        flat_features = self.encoder.get_flat_features(env)
        flat_tensor = torch.from_numpy(flat_features).unsqueeze(0).to(self.device).float()

        return board_tensor, flat_tensor

    def predict_q_values(self, board_tensor, flat_tensor):
        with torch.no_grad():
            return self.model(board_tensor, flat_tensor)