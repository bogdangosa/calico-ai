from abc import ABC, abstractmethod

class AgentBase(ABC):
    def __init__(self, config):
        self.config = config

    def get_hyperparameters(self) -> dict:
        return {}

    @abstractmethod
    def select_action(self, env):
        pass