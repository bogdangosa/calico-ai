import random

from src.agents.agent_base import AgentBase
from src.engine.scoring.scoring import ScoringCalculator
from src.engine.environments.calico_env import CalicoEnv

class OneStepLookaheadAgent(AgentBase):
    def __init__(self, scorer: ScoringCalculator, config):
        super().__init__(config)
        self.scorer = scorer

    def select_action(self, env: CalicoEnv):

        if not env.history_manager:
            env.enable_history()

        legal_actions = env.get_legal_actions()
        if not legal_actions:
            return None

        best_score = -float('inf')
        best_actions = []

        for action in legal_actions:
            env.perform_action(action)

            current_score = self.scorer.evaluate_move(
                env.board_matrix,
                env.cat_tiles
            )

            env.undo_action()

            if current_score > best_score:
                best_score = current_score
                best_actions = [action]
            elif current_score == best_score:
                best_actions.append(action)

        return random.choice(best_actions)