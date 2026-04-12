import copy
import math
import random

from src.agents.montecarlo.mcst_node import MCTSNode
from src.agents.random_agent import RandomAgent
from src.engine.environments.calico_env import CalicoEnv
from src.engine.scoring.scoring import ScoringCalculator
from src.models.game_models import CalicoAction


class MonteCarloTreeSearchAgent:
    def __init__(self, scorer: ScoringCalculator, config,exp_c=1.41,max_iterations=1000):
        self.config = config
        self.exp_c = exp_c
        self.max_iterations = max_iterations
        self.scorer = scorer

    def _select(self, node: MCTSNode) -> MCTSNode:
        """Walks down the tree using UCB until it finds a node that isn't fully expanded."""
        current_node = node
        while current_node.is_fully_expanded() and not current_node.is_terminal_node():
            current_node = current_node.get_best_ucb_child(self.exp_c)
        return current_node

    def _expand(self, node: MCTSNode) -> MCTSNode:
        """Picks one untried action, creates a new child node for it, and returns it."""
        action = random.choice(node.untried_actions)
        node.untried_actions.remove(action)

        next_state = copy.deepcopy(node.state)
        next_state.perform_action(action)

        child_node = MCTSNode(state=next_state, parent=node, action_taken=action)
        node.children.append(child_node)
        return child_node

    def _rollout(self, state) -> float:
        """Plays a completely random game from the given state to the end."""
        rollout_env = copy.deepcopy(state)
        rollout_env.disable_history()
        random_agent = RandomAgent(self.config)

        while not rollout_env.is_game_over():
            action = random_agent.select_action(rollout_env)
            rollout_env.perform_action(action)

        total_score, *_ = self.scorer.get_total_detailed_score(
            rollout_env.board_matrix,
            rollout_env.cat_tiles
        )
        return total_score

    def _backpropagate(self, node: MCTSNode, reward: float):
        """Walks back up the tree to the root, updating stats along the way."""
        current_node = node
        while current_node is not None:
            current_node.visits += 1
            current_node.total_score += reward
            current_node = current_node.parent

    def select_action(self, env: CalicoEnv):
        root_node = MCTSNode(state=copy.deepcopy(env))

        for _ in range(self.max_iterations):
            node = self._select(root_node)

            if not node.is_terminal_node() and not node.is_fully_expanded():
                node = self._expand(node)

            reward = self._rollout(node.state)

            self._backpropagate(node, reward)

        best_child = max(root_node.children, key=lambda c: c.visits)
        return best_child.action_taken