import random
from src.engine.scoring.scoring import ScoringCalculator
from src.utils.timing import time_it


class TopKLookaheadAgent:
    def __init__(self, scorer: ScoringCalculator, config, depth=2, k_factor=5):
        self.config = config
        self.depth = depth
        self.k_factor = k_factor
        self.scorer = scorer

    def _get_pruned_actions(self, env, actions):
        scored_candidates = []

        for action in actions:
            env.perform_action(action)
            score = self.scorer.evaluate_move(env.board_matrix, env.cat_tiles)
            env.undo_action()
            scored_candidates.append((score, action))

        scored_candidates.sort(key=lambda x: x[0], reverse=True)

        return [action for _, action in scored_candidates[:self.k_factor]]

    def select_action(self, env):
        legal_actions = env.get_legal_actions()
        if not legal_actions:
            return None

        top_actions = self._get_pruned_actions(env, legal_actions)

        best_score = -float('inf')
        best_moves = []

        for action in top_actions:
            env.perform_action(action)
            score = self._recursive_search(env, self.depth - 1)
            env.undo_action()

            if score > best_score:
                best_score = score
                best_moves = [action]
            elif score == best_score:
                best_moves.append(action)

        return random.choice(best_moves)

    def _recursive_search(self, env, current_depth):
        if current_depth <= 0 or env.is_game_over():
            return self.scorer.evaluate_move(env.board_matrix, env.cat_tiles)

        legal_actions = env.get_legal_actions()
        if not legal_actions:
            return self.scorer.evaluate_move(env.board_matrix, env.cat_tiles)

        pruned_actions = self._get_pruned_actions(env, legal_actions)

        max_future_score = -float('inf')
        for action in pruned_actions:
            env.perform_action(action)
            score = self._recursive_search(env, current_depth - 1)
            env.undo_action()

            if score > max_future_score:
                max_future_score = score

        return max_future_score