import os
import torch

from src.agents.q_learning.baseline_q_learning_agent import BaselineQLearningAgent
from src.agents.q_learning.q_learning_agent import QLearningAgent
from src.engine.simulator import run_simulation
from src.engine.environments.calico_env import CalicoEnv
from src.utils.config import load_config
from loguru import logger

config_path = "../../config/calico_settings.json"
model_path = "../../agent_models/full_calico/baseline_q_learning_agent_v1.0.pth"

config = load_config(config_path)
env = CalicoEnv(config)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logger.info(f"Running on device: {device}")

agent = BaselineQLearningAgent(config, model_path=model_path, epsilon=0.0, device=device)

run_simulation(
    env=env,
    agent=agent,
    num_games=100,
    progress_interval=10,
    save_to_dataset=False
)
