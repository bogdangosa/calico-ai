import copy
import math

from src.agents.random_agent import RandomAgent
from src.engine.environments.calico_env import CalicoEnv
from src.engine.scoring.scoring import ScoringCalculator
from src.models.game_models import CalicoAction


class FlatMonteCarloAgent:
    def __init__(self, scorer: ScoringCalculator, config,exp_c=1.41,max_iterations=1000):
        self.config = config
        self.exp_c = exp_c
        self.max_iterations = max_iterations
        self.scorer = scorer

    def get_hyperparameters(self):
        return {
            "max_iterations": self.max_iterations,
            "exp_c": self.exp_c,
        }

    def random_rollout(self,env: CalicoEnv):
        rollout_env = copy.deepcopy(env)
        rollout_env.disable_history()
        random_agent = RandomAgent(self.config)
        while not rollout_env.is_game_over():
            action = random_agent.select_action(rollout_env)
            if action is None:
                raise RuntimeError("No action selected.")
            rollout_env.perform_action(action)

        total,*details = self.scorer.get_total_detailed_score(rollout_env.board_matrix, rollout_env.cat_tiles)
        return total

    def calculate_ucb(self, total_simulations, total_scores_per_action, visit_counts_per_action) -> int:
        if total_simulations == 0:
            return 0

        natural_log_of_total_simulations = math.log(total_simulations)
        highest_ucb_score_found = float('-inf')
        index_of_best_action = 0

        for current_action_index, times_action_was_visited in enumerate(visit_counts_per_action):
            if times_action_was_visited == 0:
                return current_action_index

            average_score_for_action = total_scores_per_action[current_action_index] / times_action_was_visited
            exploration_bonus = self.exp_c * math.sqrt(natural_log_of_total_simulations / times_action_was_visited)
            total_ucb_score = average_score_for_action + exploration_bonus

            if total_ucb_score > highest_ucb_score_found:
                highest_ucb_score_found = total_ucb_score
                index_of_best_action = current_action_index

        return index_of_best_action

    def _best_action(self, actions, score_sums, counts) -> CalicoAction:
        best_score = float('-inf')
        best_action = None

        for i, action in enumerate(actions):
            score = (score_sums[i] / counts[i]) if counts[i] > 0 else 0.0
            if score > best_score:
                best_score = score
                best_action = action

        return best_action

    def select_action(self, env: CalicoEnv):
        if not env.history_manager:
            env.enable_history()
        legal_actions = env.get_legal_actions()
        if not legal_actions:
            return None

        total_scores_per_action = [0.0] * len(legal_actions)
        visit_counts_per_action = [0] * len(legal_actions)

        for current_iteration in range(self.max_iterations):
            target_index = self.calculate_ucb(current_iteration, total_scores_per_action, visit_counts_per_action)
            target_action = legal_actions[target_index]

            env.perform_action(target_action)
            rollout_score = self.random_rollout(env)
            env.undo_action()

            total_scores_per_action[target_index] += rollout_score
            visit_counts_per_action[target_index] += 1

        return self._best_action(legal_actions, total_scores_per_action, visit_counts_per_action)