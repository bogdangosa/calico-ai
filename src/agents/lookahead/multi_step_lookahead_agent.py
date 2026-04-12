import random

from src.engine.environments.calico_env import CalicoEnv
from src.engine.scoring.scoring import ScoringCalculator
from src.models.game_models import CalicoAction


class MultiStepLookaheadAgent:
    def __init__(self, scorer: ScoringCalculator, config, depth=1):
        self.config = config
        self.depth = depth
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

            score = self._recursive_search(env, self.depth - 1)

            env.undo_action()

            if score > best_score:
                best_score = score
                best_actions = [action]
            elif score == best_score:
                best_actions.append(action)

        return random.choice(best_actions)


    def _recursive_search(self, env, current_depth):
        if current_depth <= 0 or env.is_game_over():
            return self.scorer.evaluate_move(env.board_matrix, env.cat_tiles)

        legal_actions = env.get_legal_actions()
        if not legal_actions:
            return self.scorer.evaluate_move(env.board_matrix, env.cat_tiles)

        max_future_score = -float('inf')


        for action in legal_actions:
            env.perform_action(action)

            score = self._recursive_search(env, current_depth - 1)

            env.undo_action()

            if score > max_future_score:
                max_future_score = score

        return max_future_score