from typing import Optional, Dict, Any, Type
from src.agents.agent_base import AgentBase
from src.agents.random_agent import RandomAgent
from src.agents.lookahead.one_step_lookahead_agent import OneStepLookaheadAgent
from src.agents.lookahead.multi_step_lookahead_agent import MultiStepLookaheadAgent
from src.agents.lookahead.topk_lookahead_agent import TopKLookaheadAgent
from src.agents.montecarlo.monte_carlo_tree_search_agent import MonteCarloTreeSearchAgent
from src.agents.q_learning.q_learning_agent import QLearningAgent
from src.agents.temporal_difference.tabular_td_agent import TabularTDAgent
from src.agents.q_learning.tabular_q_learning_agent import TabularQLearningAgent
from src.agents.q_learning.baseline_q_learning_agent import BaselineQLearningAgent
from src.agents.q_learning.sb3_agent import SB3Agent
from src.engine.scoring.scoring import ScoringCalculator
from src.models.game_config import GameSettings

class AgentFactory:
    """
    Factory class to create AI agents based on strings and game configuration.
    """

    AGENT_MAP: Dict[str, Type[AgentBase]] = {
        "random": RandomAgent,
        "one_step_lookahead": OneStepLookaheadAgent,
        "multi_step_lookahead": MultiStepLookaheadAgent,
        "topk_lookahead": TopKLookaheadAgent,
        "mcts": MonteCarloTreeSearchAgent,
        "q_learning": QLearningAgent,
        "tabular_td": TabularTDAgent,
        "tabular_q_learning": TabularQLearningAgent,
        "baseline_q_learning": BaselineQLearningAgent,
        "sb3": SB3Agent
    }

    DEFAULT_CONFIGS = {
        "micro_calico": {
            "multi_step_lookahead": {"depth": 2},
            "topk_lookahead": {"depth": 3, "k_factor": 5},
            "mcts": {"max_iterations": 500, "exp_c": 1.41},
            "q_learning": {"epsilon": 0.05},
            "tabular_td": {"learning_rate": 0.1, "discount_factor": 0.95, "epsilon": 0.1},
            "tabular_q_learning": {"learning_rate": 0.1, "discount_factor": 0.95, "epsilon": 0.1},
            "baseline_q_learning": {"epsilon": 0.05}
        },
        "mini_calico": {
            "multi_step_lookahead": {"depth": 1},
            "topk_lookahead": {"depth": 2, "k_factor": 5},
            "mcts": {"max_iterations": 1000, "exp_c": 1.41},
            "q_learning": {"epsilon": 0.05},
            "tabular_td": {"learning_rate": 0.1, "discount_factor": 0.95, "epsilon": 0.1},
            "tabular_q_learning": {"learning_rate": 0.1, "discount_factor": 0.95, "epsilon": 0.1},
            "baseline_q_learning": {"epsilon": 0.05}
        },
        "main_calico": {
            "multi_step_lookahead": {"depth": 1},
            "topk_lookahead": {"depth": 2, "k_factor": 3},
            "mcts": {"max_iterations": 2000, "exp_c": 1.41},
            "q_learning": {"epsilon": 0.05},
            "tabular_td": {"learning_rate": 0.05, "discount_factor": 0.99, "epsilon": 0.1},
            "tabular_q_learning": {"learning_rate": 0.05, "discount_factor": 0.99, "epsilon": 0.1},
            "baseline_q_learning": {"epsilon": 0.05}
        }
    }

    @staticmethod
    def create_agent(
        agent_type: str, 
        game_config: GameSettings, 
        **kwargs
    ) -> AgentBase:
        """
        Creates an agent instance.
        
        Args:
            agent_type: String identifier for the agent (e.g., "random", "mcts")
            game_config: The GameSettings object containing board size, etc.
            **kwargs: Overrides for agent hyperparameters (e.g., depth=2, model_path="...")
            
        Returns:
            An instance of an AgentBase subclass.
        """
        if agent_type not in AgentFactory.AGENT_MAP:
            raise ValueError(f"Unknown agent type: {agent_type}. Available: {list(AgentFactory.AGENT_MAP.keys())}")

        agent_class = AgentFactory.AGENT_MAP[agent_type]
        game_mode = game_config.name

        agent_params = {}
        if game_mode in AgentFactory.DEFAULT_CONFIGS:
            agent_params.update(AgentFactory.DEFAULT_CONFIGS[game_mode].get(agent_type, {}))

        agent_params.update(kwargs)

        if agent_class in [OneStepLookaheadAgent, MultiStepLookaheadAgent, TopKLookaheadAgent, MonteCarloTreeSearchAgent]:
            scorer = ScoringCalculator(game_config)
            return agent_class(scorer=scorer, config=game_config, **agent_params)

        if agent_class in [QLearningAgent, TabularQLearningAgent, BaselineQLearningAgent, SB3Agent]:
            return agent_class(config=game_config, **agent_params)

        return agent_class(config=game_config, **agent_params)
