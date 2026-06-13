import torch
import os
from src.agents.montecarlo.deep_monte_carlo_tree_search_agent import DeepMCTSAgent
from src.agents.montecarlo.monte_carlo_tree_search_agent import MonteCarloTreeSearchAgent
from src.engine.scoring.potential_scoring import PotentialScoringCalculator
from src.engine.scoring.scoring import ScoringCalculator
from src.machine_learning.networks.dual_head_res_net import DualHeadResNet
from src.machine_learning.networks.dual_head_res_net2 import DualHeadResNet2
from src.engine.environments.calico_encoder import CalicoEncoder
from src.engine.environments.action_mapper import ActionMapper
from src.engine.simulator import run_simulation
from src.engine.environments.calico_env import CalicoEnv
from src.utils.config import load_config

# Paths relative to the project root
CONFIG_PATH = "../../config/calico_settings.json"
MODEL_PATH = "../../agent_models/full_calico/deep_mcts_resnet_v1.0.pth"

config = load_config(CONFIG_PATH)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

encoder = CalicoEncoder(config)
mapper = ActionMapper(
    board_size=config.board.size,
    hand_size=config.player_hand_size,
    shop_size=config.nr_of_tiles_in_shop
)

model = DualHeadResNet(
    input_channels=encoder.total_feature_layers,
    board_size=config.board.size,
    num_actions=mapper.total_actions,
    num_blocks=3,
    hidden_channels=64
).to(device)

if os.path.exists(MODEL_PATH):
    try:
        model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
        print(f"Loaded model from {MODEL_PATH}")
    except Exception as e:
        print(f"Warning: Could not load weights from {MODEL_PATH} due to architecture mismatch: {e}")
else:
    print(f"Warning: Model not found at {MODEL_PATH}. Using randomly initialized weights.")

model.eval()

agent_deep = DeepMCTSAgent(
    config=config,
    network=model,
    encoder=encoder,
    mapper=mapper,
    num_simulations=50,
    c_puct=1.0
)

scorer = PotentialScoringCalculator(config)
agent = MonteCarloTreeSearchAgent(scorer,config,max_iterations=50)

env = CalicoEnv(config)

run_simulation(
    env=env,
    agent=agent_deep,
    num_games=100,
    progress_interval=5,
    save_to_dataset=True
)
