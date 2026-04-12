import math


class MCTSNode:
    def __init__(self, state, parent=None, action_taken=None):
        self.state = state
        self.parent = parent
        self.action_taken = action_taken
        self.children = []
        self.visits = 0
        self.total_score = 0.0

        self.untried_actions = state.get_legal_actions() if not state.is_game_over() else []

    def is_fully_expanded(self) -> bool:
        return len(self.untried_actions) == 0

    def is_terminal_node(self) -> bool:
        return self.state.is_game_over()

    def get_best_ucb_child(self, exploration_constant: float):
        best_score = float('-inf')
        best_child = None

        for child in self.children:
            exploitation = child.total_score / child.visits
            exploration = exploration_constant * math.sqrt(math.log(self.visits) / child.visits)
            ucb_score = exploitation + exploration

            if ucb_score > best_score:
                best_score = ucb_score
                best_child = child

        return best_child