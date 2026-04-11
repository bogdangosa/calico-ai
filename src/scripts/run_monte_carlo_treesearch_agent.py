from src.agents.montecarlo.flat_monte_carlo_agent import FlatMonteCarloAgent
from src.agents.montecarlo.monte_carlo_tree_search_agent import MonteCarloTreeSearchAgent
from src.engine.scoring.potential_scoring import PotentialScoringCalculator
from src.engine.simulator import run_simulation
from src.engine.environments.calico_env import CalicoEnv
from src.utils.config import load_config

config = load_config("../../config/calico_settings.json")
env = CalicoEnv(config)
scorer = PotentialScoringCalculator(config)
agent = MonteCarloTreeSearchAgent(scorer,config,max_iterations=100)

run_simulation(
    env=env,
    agent=agent,
    num_games=1,
    progress_interval=1
)