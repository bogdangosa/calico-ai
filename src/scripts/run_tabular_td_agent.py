import os

from src.agents.q_learning.tabular_q_learning_agent import TabularQLearningAgent
from src.agents.temporal_difference.tabular_td_agent import TabularTDAgent
from src.engine.simulator import run_simulation
from src.engine.environments.calico_env import CalicoEnv
from src.utils.config import load_config
from loguru import logger

# Paths relative to the project root
config_path = "../../config/micro_calico_settings_v2.json"
v_table_path = "../../agent_models/micro_calico_v2/tabular_v_table_v2.0.pkl"

def run_evaluation():
    config = load_config(config_path)
    env = CalicoEnv(config)

    # Initialize agent with epsilon=0 for pure exploitation
    agent = TabularTDAgent(config, epsilon=0.0)
    
    if os.path.exists(v_table_path):
        logger.info(f"Loading V-table from {v_table_path}")
        agent.load(v_table_path)
    else:
        logger.warning(f"No V-table found at {v_table_path}. Running with empty table.")

    logger.info(f"Running evaluation on {config.name}")

    # To see the total number of unique state-action pairs discovered
    print(f"Total Q-table Entries: {len(agent.v_table)}")

    # To inspect a few raw keys and values directly
    for key, value in list(agent.v_table.items())[:3]:
        print(f"Key: {key} -> Expected Score: {value}")

    run_simulation(
        env=env,
        agent=agent,
        num_games=1000,
        progress_interval=10,
        save_to_dataset=False
    )

if __name__ == "__main__":
    run_evaluation()
