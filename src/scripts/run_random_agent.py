from src.engine.simulator import run_simulation
from src.engine.environments.calico_env import CalicoEnv
from src.agents.random_agent import RandomAgent
from src.utils.config import load_config

config = load_config("../../config/calico_settings.json")
env = CalicoEnv(config)
agent = RandomAgent(config)

run_simulation(
    env=env,
    agent=agent,
    num_games=10000,
    progress_interval=100,
    save_to_dataset=False
)