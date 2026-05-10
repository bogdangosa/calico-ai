import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from typing import List, Tuple, Dict

from src.agents.montecarlo.mcst_node import DeepMCTSNode
from src.engine.environments.action_mapper import ActionMapper
from src.engine.environments.calico_encoder import CalicoEncoder
from src.models.game_models import CalicoAction, ActionType
from src.engine.environments.calico_env import CalicoEnv
from src.agents.agent_base import AgentBase
from src.engine.scoring.scoring import ScoringCalculator

class DeepMCTSAgent(AgentBase):
    def __init__(self, config, network: nn.Module, encoder: CalicoEncoder, mapper: ActionMapper, num_simulations: int = 100, c_puct: float = 1.0):
        super().__init__(config)
        self.network = network
        self.encoder = encoder
        self.mapper = mapper
        self.num_simulations = num_simulations
        self.c_puct = c_puct
        self.network.eval()

    def select_action(self, env: CalicoEnv) -> CalicoAction:
        env.enable_history()
        root = DeepMCTSNode(prior=1.0)
        self._expand_node(root, env)

        for _ in range(self.num_simulations):
            node = root
            search_path: List[Tuple[DeepMCTSNode, int]] = []
            
            while node.children:
                best_action, next_node = self._select_child(node)
                env.perform_action(self.mapper.index_to_action(best_action))
                search_path.append((node, best_action))
                node = next_node

            if not env.is_game_over():
                value = self._expand_node(node, env)
            else:
                scorer = ScoringCalculator(env.config)
                value = scorer.evaluate_move(env.board_matrix, env.cat_tiles)

            self._backpropagate(search_path, node, value)

            for _ in search_path:
                env.undo_action()

        env.disable_history()
        best_visit_count = -1
        best_action_index = -1
        
        for action_index, child in root.children.items():
            if child.visit_count > best_visit_count:
                best_visit_count = child.visit_count
                best_action_index = action_index

        return self.mapper.index_to_action(best_action_index)

    def _select_child(self, node: DeepMCTSNode) -> Tuple[int, DeepMCTSNode]:
        parent_visit_count = node.visit_count
        action_indices = np.array(list(node.children.keys()))
        q_values = np.zeros(len(action_indices))
        priors = np.zeros(len(action_indices))
        child_visits = np.zeros(len(action_indices))

        for idx, act in enumerate(action_indices):
            child = node.children[act]
            q_values[idx] = child.q_value
            priors[idx] = child.prior
            child_visits[idx] = child.visit_count

        u_values = q_values + self.c_puct * priors * (np.sqrt(parent_visit_count) / (1.0 + child_visits))
        
        best_idx = int(np.argmax(u_values))
        best_action = int(action_indices[best_idx])
        
        return best_action, node.children[best_action]

    def _expand_node(self, node: DeepMCTSNode, env: CalicoEnv) -> float:
        legal_actions = env.get_legal_actions()
        if not legal_actions:
            return 0.0

        state_tensor = self.encoder.encode(env)
        with torch.no_grad():
            policy_probs, value = self.network(state_tensor)

        policy_probs = policy_probs.squeeze(0).cpu().numpy()
        value = value.item()

        legal_indices = [self.mapper.action_to_index(a) for a in legal_actions]
        legal_probs = policy_probs[legal_indices]
        sum_probs = np.sum(legal_probs)
        if sum_probs > 0:
            legal_probs /= sum_probs
        else:
            legal_probs = np.ones(len(legal_indices)) / len(legal_indices)

        for act_idx, prob in zip(legal_indices, legal_probs):
            node.children[act_idx] = DeepMCTSNode(prior=prob)

        return value

    def _backpropagate(self, search_path: List[Tuple[DeepMCTSNode, int]], leaf_node: DeepMCTSNode, value: float) -> None:
        leaf_node.visit_count += 1
        leaf_node.total_value += value
        for node, _ in reversed(search_path):
            node.visit_count += 1
            node.total_value += value


class SelfPlayRunner:
    def __init__(self, env_config, agent: DeepMCTSAgent):
        self.env_config = env_config
        self.agent = agent

    def play_game(self) -> List[Tuple[np.ndarray, np.ndarray, float]]:
        env = CalicoEnv(self.env_config)
        env.start_game()
        
        trajectory = []
        
        while not env.is_game_over():
            root = DeepMCTSNode(prior=1.0)
            
            env.enable_history()
            self.agent._expand_node(root, env)

            for _ in range(self.agent.num_simulations):
                node = root
                search_path = []
                
                while node.children:
                    best_action, next_node = self.agent._select_child(node)
                    env.perform_action(self.agent.mapper.index_to_action(best_action))
                    search_path.append((node, best_action))
                    node = next_node

                if not env.is_game_over():
                    value = self.agent._expand_node(node, env)
                else:
                    scorer = ScoringCalculator(env.config)
                    value = scorer.evaluate_move(env.board_matrix, env.cat_tiles)

                self.agent._backpropagate(search_path, node, value)

                for _ in search_path:
                    env.undo_action()

            env.disable_history()
            
            visit_counts = np.zeros(self.agent.mapper.total_actions)
            for action_index, child in root.children.items():
                visit_counts[action_index] = child.visit_count
                
            visit_probs = visit_counts / np.sum(visit_counts) if np.sum(visit_counts) > 0 else visit_counts
            
            best_action_index = int(np.argmax(visit_counts))
            best_action = self.agent.mapper.index_to_action(best_action_index)
            
            state_data = self.agent.encoder.encode(env).squeeze(0).cpu().numpy()
            trajectory.append((state_data, visit_probs))
            
            env.perform_action(best_action)

        scorer = ScoringCalculator(env.config)
        final_score = scorer.evaluate_move(env.board_matrix, env.cat_tiles)
        
        result_data = []
        for state, probs in trajectory:
            result_data.append((state, probs, final_score))
            
        return result_data


class MCTSTrainer:
    def __init__(self, network: nn.Module, lr: float = 1e-3, weight_decay: float = 1e-4):
        self.network = network
        self.optimizer = optim.Adam(self.network.parameters(), lr=lr, weight_decay=weight_decay)

    def train_step(self, states: torch.Tensor, target_probs: torch.Tensor, target_values: torch.Tensor) -> float:
        self.network.train()
        self.optimizer.zero_grad()
        
        pred_probs, pred_values = self.network(states)
        pred_values = pred_values.squeeze(-1)
        
        value_loss = F.mse_loss(pred_values, target_values)
        policy_loss = -torch.sum(target_probs * torch.log(pred_probs + 1e-8), dim=1).mean()
        
        total_loss = value_loss + policy_loss
        total_loss.backward()
        self.optimizer.step()
        
        return total_loss.item()
