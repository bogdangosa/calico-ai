import json

from src.engine.simulator import run_simulation
from src.models.game_config import GameSettings
from src.engine.environments.calico_env import CalicoEnv
from src.agents.random_agent import RandomAgent

config_path = "../../config/calico_settings.json"
with open(config_path, "r") as f:
    config_data = json.load(f)

config = GameSettings(**config_data)
env = CalicoEnv(config)
agent = RandomAgent(config)

run_simulation(
    env=env,
    agent=agent,
    num_games=10000,
    progress_interval=1000
)