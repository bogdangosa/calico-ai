import math
import random
import copy
import numpy as np

from enviroment.calico_env import CalicoEnv
from enviroment.calico_scoring import get_total_score_on_board


class MCTSNode:
    def __init__(self, state, parent=None, action=None):
        self.state = copy.deepcopy(state)
        self.parent = parent
        self.action = action
        self.children = []
        self.visits = 0
        self.reward = 0.0

    def is_fully_expanded(self):
        return len(self.children) == len(self.state.get_legal_actions())

    def best_child(self, c_param=1.4):
        choices = [
            (child.reward / (child.visits + 1e-5) +
             c_param * math.sqrt(math.log(self.visits + 1) / (child.visits + 1e-5)), child)
            for child in self.children
        ]
        return max(choices, key=lambda x: x[0])[1]

def mcts(env,n_simulations=1000):
    root = MCTSNode(env)

    for i in range(n_simulations):
        node = root
        sim_env = copy.deepcopy(env)

        # Selection
        while node.children and node.is_fully_expanded():
            node = node.best_child()
            sim_env.perform_action(node.action)

        # Expansion
        legal_actions = sim_env.get_legal_actions()
        untried_actions = [a for a in legal_actions if a not in [c.action for c in node.children]]
        if untried_actions:
            action = random.choice(untried_actions)
            sim_env.perform_action(action)
            child_node = MCTSNode(sim_env, parent=node, action=action)
            node.children.append(child_node)
            node = child_node

        # Simulation
        sim2_env = copy.deepcopy(sim_env)
        reward = rollout(sim2_env)

        # Backpropagation
        while node:
            node.visits += 1
            node.reward += reward
            node = node.parent
        #print(f"pass {i} reward {reward}")
    # Choose the most visited child as the action
    best_child_node = max(root.children, key=lambda c: c.visits)
    return best_child_node.action

def rollout(env):
    """Random rollout until game over"""
    env.start_game()
    while not env.is_game_over():
        legal_actions = env.get_legal_actions()
        action = random.choice(legal_actions)
        env.perform_action(action)
    return get_total_score_on_board(env.board_matrix, env.cat_tiles)

env = CalicoEnv()
env.start_game()

while not env.is_game_over():
    # Get best action from MCTS using current state
    best_action = mcts(env, n_simulations=500)  # for example 500 simulations
    print(best_action)
    print("score"+str(get_total_score_on_board(env.board_matrix, env.cat_tiles)))

    # Perform that action
    env.perform_action(best_action)