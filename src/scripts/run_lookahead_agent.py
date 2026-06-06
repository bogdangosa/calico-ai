
from src.agents.lookahead.multi_step_lookahead_agent import MultiStepLookaheadAgent
from src.agents.lookahead.one_step_lookahead_agent import OneStepLookaheadAgent
from src.agents.lookahead.topk_lookahead_agent import TopKLookaheadAgent
from src.engine.scoring.potential_scoring import PotentialScoringCalculator
from src.engine.scoring.scoring import ScoringCalculator
from src.engine.simulator import run_simulation
from src.engine.environments.calico_env import CalicoEnv
from src.utils.config import load_config

config = load_config("../../config/calico_settings.json")
env = CalicoEnv(config)
scorer = PotentialScoringCalculator(config)
agent = OneStepLookaheadAgent(scorer,config)

run_simulation(
    env=env,
    agent=agent,
    num_games=1000,
    progress_interval=10,
    save_to_dataset=True
)