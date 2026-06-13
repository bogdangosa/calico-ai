import random

from src.agents.agent_base import AgentBase
from src.engine.environments.calico_env import CalicoEnv


class RandomAgent(AgentBase):
    def __init__(self, config):
        super().__init__(config)

    def get_hyperparameters(self):
        return {}

    def select_action(self, env: CalicoEnv):
        legal_actions = env.get_legal_actions()
        if not legal_actions:
            return None
        return random.choice(legal_actions)