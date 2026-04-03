import json

from src.agents.one_step_lookahead_agent import OneStepLookaheadAgent
from src.engine.scoring.potential_scoring import PotentialScoringCalculator
from src.engine.scoring.scoring import ScoringCalculator
from src.engine.simulator import run_simulation
from src.models.game_config import GameSettings
from src.engine.environments.calico_env import CalicoEnv

config_path = "../../config/calico_settings.json"
with open(config_path, "r") as f:
    config_data = json.load(f)

config = GameSettings(**config_data)
env = CalicoEnv(config)
scorer = PotentialScoringCalculator(config)
agent = OneStepLookaheadAgent(scorer,config)

run_simulation(
    env=env,
    agent=agent,
    num_games=100,
    progress_interval=10
)