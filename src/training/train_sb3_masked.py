import os
from datetime import datetime
from sb3_contrib import MaskablePPO
from sb3_contrib.common.maskable.utils import get_action_masks
from sb3_contrib.common.wrappers import ActionMasker
from stable_baselines3.common.callbacks import CheckpointCallback
from loguru import logger

from src.engine.environments.calico_gym_wrapper import CalicoGymWrapper
from src.utils.config import load_config

# Configuration
CONFIG_PATH = "../../config/calico_settings.json"
TOTAL_TIMESTEPS = 5000000
MODEL_SAVE_PATH = "../../agent_models/full_calico/sb3_masked_ppo"
LOG_DIR = f"../../outputs/logs/sb3_training/{datetime.now().strftime('%Y%m%d-%H%M%S')}"

def mask_fn(env: CalicoGymWrapper) -> bytes:
    """Standard mask function for ActionMasker."""
    return env.action_masks()

def train():
    # 1. Load configuration and create environment
    config = load_config(CONFIG_PATH)
    raw_env = CalicoGymWrapper(config)
    
    # 2. Wrap the environment with ActionMasker for SB3-contrib compatibility
    env = ActionMasker(raw_env, mask_fn)
    
    # 3. Initialize the MaskablePPO model
    model = MaskablePPO(
        "MultiInputPolicy", 
        env, 
        verbose=1, 
        tensorboard_log=LOG_DIR,
        learning_rate=1e-4,
        gamma=0.98,
        batch_size=256,
        n_steps=2048
    )
    
    # 4. Setup callbacks
    checkpoint_callback = CheckpointCallback(
        save_freq=100000,
        save_path=MODEL_SAVE_PATH,
        name_prefix="sb3_calico_ppo"
    )
    
    # 5. Train the model
    logger.info(f"Starting SB3 MaskedPPO training for {TOTAL_TIMESTEPS} timesteps...")
    model.learn(
        total_timesteps=TOTAL_TIMESTEPS,
        callback=checkpoint_callback,
        progress_bar=True
    )
    
    # 6. Save final model
    os.makedirs(MODEL_SAVE_PATH, exist_ok=True)
    final_path = os.path.join(MODEL_SAVE_PATH, "final_model")
    model.save(final_path)
    logger.info(f"Training complete. Model saved to {final_path}")
    
    # 7. Basic Evaluation
    logger.info("Running a few evaluation games...")
    obs, info = env.reset()
    total_rewards = 0
    num_eval_games = 5
    games_played = 0
    
    while games_played < num_eval_games:
        # Retrieve current action mask from the wrapped environment
        action_masks = get_action_masks(env)
        action, _states = model.predict(obs, action_masks=action_masks, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)
        total_rewards += reward
        
        if terminated or truncated:
            games_played += 1
            logger.info(f"Game {games_played} finished. Final Score: {info['score']}")
            obs, info = env.reset()
            
    logger.info(f"Average Reward over {num_eval_games} games: {total_rewards/num_eval_games:.2f}")

if __name__ == "__main__":
    train()
