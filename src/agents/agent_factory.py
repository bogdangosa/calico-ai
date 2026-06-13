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
from src.engine.scoring.potential_scoring import PotentialScoringCalculator
from src.engine.scoring.scoring import ScoringCalculator
from src.models.game_config import GameSettings

class AgentFactory:

    AGENT_MAP: Dict[str, Type[AgentBase]] = {
        "Random Agent": RandomAgent,
        "OneStepLookahead Agent": OneStepLookaheadAgent,
        "MultiStepLookahead Agent": MultiStepLookaheadAgent,
        "TopKLookahead Agent": TopKLookaheadAgent,
        "TabularTD Agent": TabularTDAgent,
        "TabularQLearning Agent": TabularQLearningAgent,
        "QLearning Agent": BaselineQLearningAgent,
        "SB3 Agent": SB3Agent
    }

    DEFAULT_CONFIGS = {
        "micro_calico": {
            "MultiStepLookahead Agent": {"depth": 3},
            "TopKLookahead Agent": {"depth": 3, "k_factor": 5},
            "TabularTD Agent": {"learning_rate": 0.1, "discount_factor": 0.95, "epsilon": 0},
            "TabularQLearning Agent": {"learning_rate": 0.1, "discount_factor": 0.95, "epsilon": 0},
            "QLearning Agent": {"epsilon": 0.05}
        },
        "main_calico": {
            "MultiStepLookahead Agent": {"depth": 3},
            "SB3 Agent": {"model_path":"agent_models/full_calico/sb3_full_calico_ppo_35mil.zip"},
            "TopKLookahead Agent": {"depth": 2, "k_factor": 3},
            "QLearning Agent": {"epsilon": 0,"model_path":"agent_models/full_calico/baseline_q_terminal_anchoring_v1.3.pth"}
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
            scorer = PotentialScoringCalculator(game_config)
            return agent_class(scorer=scorer, config=game_config, **agent_params)

        if agent_class in [QLearningAgent, TabularQLearningAgent, BaselineQLearningAgent, SB3Agent]:
            return agent_class(config=game_config, **agent_params)

        return agent_class(config=game_config, **agent_params)
