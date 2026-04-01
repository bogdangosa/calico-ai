import random

class RandomAgent:
    def __init__(self, config):
        self.config = config

    def select_action(self, env):
        """Standard interface for all future agents."""
        legal_actions = env.get_legal_actions()
        if not legal_actions:
            return None
        return random.choice(legal_actions)