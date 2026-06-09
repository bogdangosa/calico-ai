import numpy as np
import torch
from typing import Optional, List
from src.agents.agent_base import AgentBase
from src.engine.environments.calico_gym_wrapper import CalicoGymWrapper
from src.models.game_models import CalicoAction

class SB3Agent(AgentBase):
    """
    Agent that uses a trained Stable-Baselines3 model.
    It utilizes the CalicoGymWrapper for state conversion and action mapping.
    """
    def __init__(self, config, model_path: Optional[str] = None, device: str = "cpu"):
        super().__init__(config)
        self.device = torch.device(device)
        self.wrapper = CalicoGymWrapper(config)
        self.model = None
        
        if model_path:
            self.load(model_path)

    def select_action(self, env) -> Optional[CalicoAction]:
        if self.model is None:
            # Fallback to random if no model loaded
            legal_actions = env.get_legal_actions()
            return np.random.choice(legal_actions) if legal_actions else None

        # Synchronize wrapper's internal env state with the provided env
        # This is slightly inefficient but ensures consistency if we are using the factory
        self.wrapper.env = env
        
        # Get observation and mask
        action_mask = self.wrapper.action_masks()
        
        obs_dict = {
            "board": self.wrapper.encoder.encode(env),
            "flat_features": self.wrapper.get_flat_features(),
            "action_mask": action_mask
        }
        
        # Predict action index using the model and mask
        action_idx, _states = self.model.predict(
            obs_dict, 
            action_masks=action_mask, 
            deterministic=True
        )
        
        # Map back to CalicoAction
        return self.wrapper._map_index_to_action(int(action_idx))

    def load(self, path: str):
        """Loads a MaskablePPO model from the given path."""
        try:
            from sb3_contrib import MaskablePPO
            self.model = MaskablePPO.load(path, device=self.device)
        except ImportError:
            print("Error: sb3-contrib not installed. Cannot load MaskablePPO model.")
        except Exception as e:
            print(f"Error loading SB3 model: {e}")

    def save(self, path: str):
        """Saves the internal SB3 model."""
        if self.model:
            self.model.save(path)
