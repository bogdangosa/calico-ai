import torch
import numpy as np
import random
from src.agents.agent_base import AgentBase
from src.machine_learning.networks.baseline_q_network import BaselineQNetwork
from src.machine_learning.networks.q_network import QValueNetwork
from src.engine.environments.calico_encoder import CalicoEncoder
from src.models.game_models import CalicoAction

class BaselineQLearningAgent(AgentBase):
    """
    An agent that uses a Q-Value Network to select actions.
    It performs a one-step lookahead to evaluate the resulting state of each legal action.
    """
    def __init__(self, config, model_path=None, epsilon=0.1, device="cpu"):
        super().__init__(config)
        self.encoder = CalicoEncoder(config)
        self.device = torch.device(device)
        self.model = BaselineQNetwork(
            input_channels=self.encoder.total_feature_layers,
            board_size=config.board.size
        ).to(self.device)
        
        if model_path:
            self.model.load(model_path)
            
        self.epsilon = epsilon
        self.model.eval()

    def select_action(self, env) -> CalicoAction:
        """
        Selects an action using an epsilon-greedy policy.
        """
        legal_actions = env.get_legal_actions()
        if not legal_actions:
            return None
            
        if random.random() < self.epsilon:
            return random.choice(legal_actions)
            
        return self._get_best_action(env, legal_actions)

    def _get_best_action(self, env, legal_actions):
        """
        Evaluates all legal actions using one-step lookahead and returns the best one.
        """
        best_value = -float('inf')
        best_action = None

        history_was_enabled = env.history_manager is not None
        if not history_was_enabled:
            env.enable_history()
            
        for action in legal_actions:
            env.perform_action(action)
            state_tensor = self.get_state_tensor(env)
            with torch.no_grad():
                value = self.model(state_tensor).item()
            env.undo_action()
            
            if value > best_value:
                best_value = value
                best_action = action
        
        if not history_was_enabled:
            env.disable_history()
                
        return best_action

    def get_state_tensor(self, env):
        """
        Encodes the environment state into a PyTorch tensor ready for the network.
        """
        encoded = self.encoder.encode(env)
        encoded = np.transpose(encoded, (2, 0, 1))
        tensor = torch.from_numpy(encoded).unsqueeze(0).to(self.device)
        return tensor

    def predict_value(self, state_tensor):
        """
        Predicts the value of a given state tensor.
        """
        with torch.no_grad():
            return self.model(state_tensor)
