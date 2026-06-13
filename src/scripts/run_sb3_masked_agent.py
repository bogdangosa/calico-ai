import os
from loguru import logger

from src.agents.q_learning.sb3_agent import SB3Agent
from src.utils.config import load_config
from src.agents.agent_factory import AgentFactory
from src.engine.environments.calico_env import CalicoEnv
from src.engine.simulator import run_simulation

# Configuration
CONFIG_PATH = "../../config/calico_settings.json"
MODEL_PATH = "../../agent_models/full_calico/sb3_full_calico_ppo_40mil.zip"
NUM_GAMES = 100

def run_evaluation():
    # 1. Load configuration
    config = load_config(CONFIG_PATH)
    env = CalicoEnv(config)
    
    # 2. Check if model exists
    if not os.path.exists(MODEL_PATH):
        logger.error(f"Model not found at {MODEL_PATH}. Please run training first.")
        return

    # 3. Create SB3 Agent via Factory
    logger.info(f"Loading SB3 Masked PPO agent from {MODEL_PATH}")
    agent = SB3Agent(
        config,
        model_path=MODEL_PATH
    )

    # 4. Run Simulation
    logger.info(f"Running evaluation of {NUM_GAMES} games...")
    scores = run_simulation(
        env=env,
        agent=agent,
        num_games=NUM_GAMES,
        progress_interval=100,
        save_to_dataset=True
    )
    
    logger.info(f"Evaluation complete.")
    logger.info(f"Average Score: {sum(scores)/len(scores):.2f}")
    logger.info(f"Max Score: {max(scores)}")
    logger.info(f"Min Score: {min(scores)}")

if __name__ == "__main__":
    run_evaluation()
