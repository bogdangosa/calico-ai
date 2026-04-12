
from src.agents.lookahead.multi_step_lookahead_agent import MultiStepLookaheadAgent
from src.engine.scoring.potential_scoring import PotentialScoringCalculator
from src.engine.simulator import run_simulation
from src.engine.environments.calico_env import CalicoEnv
from src.utils.config import load_config

config = load_config("../../config/calico_settings.json")
env = CalicoEnv(config)
scorer = PotentialScoringCalculator(config)
agent = MultiStepLookaheadAgent(scorer,config,depth=2)

run_simulation(
    env=env,
    agent=agent,
    num_games=100,
    progress_interval=10
)