
from src.agents.multi_step_lookahead_agent import MultiStepLookaheadAgent
from src.agents.one_step_lookahead_agent import OneStepLookaheadAgent
from src.agents.topk_lookahead_agent import TopKLookaheadAgent
from src.engine.scoring.potential_scoring import PotentialScoringCalculator
from src.engine.scoring.scoring import ScoringCalculator
from src.engine.simulator import run_simulation
from src.models.game_config import GameSettings
from src.engine.environments.calico_env import CalicoEnv
from src.utils.config import load_config

config = load_config("../../config/calico_settings.json")
env = CalicoEnv(config)
scorer = PotentialScoringCalculator(config)
agent = MultiStepLookaheadAgent(scorer,config,depth=1)

run_simulation(
    env=env,
    agent=agent,
    num_games=100,
    progress_interval=10
)