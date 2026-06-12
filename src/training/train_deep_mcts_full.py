import os
import random
import torch
import numpy as np
from tqdm import tqdm
from loguru import logger
from collections import deque
from torch.utils.tensorboard import SummaryWriter

from src.utils.config import load_config
from src.engine.environments.calico_env import CalicoEnv
from src.engine.environments.calico_encoder import CalicoEncoder
from src.engine.environments.action_mapper import ActionMapper
from src.machine_learning.networks.dual_head_res_net2 import DualHeadResNet2
from src.agents.montecarlo.deep_monte_carlo_tree_search_agent import DeepMCTSAgent, SelfPlayRunner, MCTSTrainer

# Hyperparameters for Full Calico
CONFIG_PATH = "../../config/calico_settings.json"
MODEL_SAVE_PATH = "../../agent_models/full_calico/deep_mcts_resnet_v1.0.pth"
LOG_DIR = "../../outputs/logs/deep_mcts_full_calico"
NUM_GENERATIONS = 1000
GAMES_PER_GENERATION = 20
TRAIN_STEPS_PER_GENERATION = 50
BATCH_SIZE = 64
REPLAY_BUFFER_SIZE = 50000
NUM_SIMULATIONS = 100 
LR = 5e-4
WEIGHT_DECAY = 1e-4
SCORE_NORMALIZATION = 50.0

class ReplayBuffer:
    def __init__(self, capacity: int):
        self.buffer = deque(maxlen=capacity)

    def add(self, trajectory):
        for item in trajectory:
            self.buffer.append(item)

    def sample(self, batch_size: int):
        batch = random.sample(self.buffer, min(len(self.buffer), batch_size))
        states, flat_features, probs, values = zip(*batch)
        return (
            torch.from_numpy(np.array(states)).float(),
            torch.from_numpy(np.array(flat_features)).float(),
            torch.from_numpy(np.array(probs)).float(),
            torch.from_numpy(np.array(values)).float()
        )

    def __len__(self):
        return len(self.buffer)

def main():
    config = load_config(CONFIG_PATH)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    os.makedirs(os.path.dirname(MODEL_SAVE_PATH), exist_ok=True)
    writer = SummaryWriter(LOG_DIR)

    encoder = CalicoEncoder(config)
    mapper = ActionMapper(
        board_size=config.board.size,
        hand_size=config.player_hand_size,
        shop_size=config.nr_of_tiles_in_shop
    )

    # Full calico board size 7, 13 input channels
    model = DualHeadResNet2(
        input_channels=encoder.total_feature_layers,
        board_size=config.board.size,
        num_actions=mapper.total_actions,
        flat_features_size=encoder.flat_features_size,
        num_blocks=3,
        hidden_channels=64
    ).to(device)

    if os.path.exists(MODEL_SAVE_PATH):
        logger.info(f"Loading existing model from {MODEL_SAVE_PATH}")
        try:
            model.load_state_dict(torch.load(MODEL_SAVE_PATH, map_location=device))
        except Exception as e:
            logger.warning(f"Could not load checkpoint due to architecture mismatch: {e}. Starting fresh.")

    agent = DeepMCTSAgent(config, model, encoder, mapper, 
                          num_simulations=NUM_SIMULATIONS, 
                          score_normalization=SCORE_NORMALIZATION)
    runner = SelfPlayRunner(config, agent)
    trainer = MCTSTrainer(model, lr=LR, weight_decay=WEIGHT_DECAY)
    replay_buffer = ReplayBuffer(REPLAY_BUFFER_SIZE)

    for gen in range(NUM_GENERATIONS):
        logger.info(f"--- Generation {gen + 1}/{NUM_GENERATIONS} ---")
        
        # Self-play phase
        logger.info(f"Starting self-play ({GAMES_PER_GENERATION} games)...")
        model.eval()
        generation_scores = []
        for _ in tqdm(range(GAMES_PER_GENERATION)):
            trajectory = runner.play_game()
            replay_buffer.add(trajectory)
            generation_scores.append(trajectory[-1][3])
        
        avg_score = np.mean(generation_scores) * agent.score_normalization
        logger.info(f"Self-play finished. Avg score: {avg_score:.2f}")
        writer.add_scalar("SelfPlay/AverageScore", avg_score, gen)

        # Training phase
        if len(replay_buffer) >= BATCH_SIZE:
            logger.info(f"Starting training phase ({TRAIN_STEPS_PER_GENERATION} steps)...")
            model.train()
            total_losses = []
            policy_losses = []
            value_losses = []
            for _ in range(TRAIN_STEPS_PER_GENERATION):
                states, flat_features, probs, values = replay_buffer.sample(BATCH_SIZE)
                states, flat_features, probs, values = states.to(device), flat_features.to(device), probs.to(device), values.to(device)
                loss, p_loss, v_loss = trainer.train_step(states, flat_features, probs, values)
                total_losses.append(loss)
                policy_losses.append(p_loss)
                value_losses.append(v_loss)
            
            avg_loss = np.mean(total_losses)
            avg_p_loss = np.mean(policy_losses)
            avg_v_loss = np.mean(value_losses)
            logger.info(f"Training finished. Avg loss: {avg_loss:.4f} (P: {avg_p_loss:.4f}, V: {avg_v_loss:.4f})")
            
            writer.add_scalar("Train/TotalLoss", avg_loss, gen)
            writer.add_scalar("Train/PolicyLoss", avg_p_loss, gen)
            writer.add_scalar("Train/ValueLoss", avg_v_loss, gen)

        # Save model
        torch.save(model.state_dict(), MODEL_SAVE_PATH)
        
        # Periodic checkpoint
        if (gen + 1) % 50 == 0:
            checkpoint_path = MODEL_SAVE_PATH.replace(".pth", f"_gen_{gen+1}.pth")
            torch.save(model.state_dict(), checkpoint_path)
            logger.info(f"Checkpoint saved to {checkpoint_path}")

    writer.close()

if __name__ == "__main__":
    main()
