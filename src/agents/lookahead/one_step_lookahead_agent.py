import random

from src.engine.scoring.scoring import ScoringCalculator
from src.engine.environments.calico_env import CalicoEnv

class OneStepLookaheadAgent:
    def __init__(self, scorer: ScoringCalculator, config):
        self.config = config
        self.scorer = scorer

    def get_hyperparameters(self):
        return {}

    def select_action(self, env: CalicoEnv):
        """
        Evaluates all legal actions by simulating them and picking the one
        that results in the highest immediate score/potential.
        """
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