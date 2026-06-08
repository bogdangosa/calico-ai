import os
from datetime import datetime
from sb3_contrib import MaskablePPO
from sb3_contrib.common.maskable.utils import get_action_masks
from sb3_contrib.common.wrappers import ActionMasker
from stable_baselines3.common.callbacks import CheckpointCallback, BaseCallback
from loguru import logger

from src.engine.environments.calico_gym_wrapper import CalicoGymWrapper
from src.utils.config import load_config

# Configuration for FULL CALICO (7x7 Board)
CONFIG_PATH = "../../config/calico_settings.json"
TOTAL_TIMESTEPS = 20000000  # Another 11 Million steps
MODEL_SAVE_PATH = "../../agent_models/full_calico/sb3_masked_ppo_v3.1"
LOAD_MODEL_PATH = "../../agent_models/full_calico/sb3_masked_ppo/sb3_calico_ppo_39mil_v1.2.zip"
LOG_DIR = f"../../outputs/logs/sb3_training_full/{datetime.now().strftime('%Y%m%d-%H%M%S')}"

class ActualScoreCallback(BaseCallback):
    def _on_step(self) -> bool:
        if self.locals.get("dones", [False])[0]:
            final_info = self.locals["infos"][0]
            if "score" in final_info:
                self.logger.record("metrics/actual_calico_score", final_info["score"])
        return True


def mask_fn(env: CalicoGymWrapper) -> bytes:
    """Standard mask function for ActionMasker."""
    return env.action_masks()

def train():
    # 1. Load configuration and create environment
    if not os.path.exists(CONFIG_PATH):
        alt_path = "../../" + CONFIG_PATH
        if os.path.exists(alt_path):
            config = load_config(alt_path)
        else:
            # Try absolute path or relative to project root
            # In some execution contexts, paths might differ
            config = load_config("config/calico_settings.json")
    else:
        config = load_config(CONFIG_PATH)
        
    raw_env = CalicoGymWrapper(config)
    env = ActionMasker(raw_env, mask_fn)
    
    # 3. Load or Initialize the MaskablePPO model
    logger.info(f"Loading existing model from {LOAD_MODEL_PATH}...")
    model = MaskablePPO.load(
        LOAD_MODEL_PATH, 
        env=env, 
        tensorboard_log=LOG_DIR,
    )
    
    # 4. Setup callbacks
    checkpoint_callback = CheckpointCallback(
        save_freq=1000000, # Save every 1M steps
        save_path=MODEL_SAVE_PATH,
        name_prefix="sb3_full_calico_ppo_30mil_checkpoint"
    )

    score_logger = ActualScoreCallback()

    # 5. Train the model
    logger.info(f"Starting SB3 Full Calico training for another {TOTAL_TIMESTEPS} timesteps...")
    model.learn(
        total_timesteps=TOTAL_TIMESTEPS,
        callback=[checkpoint_callback, score_logger],
        progress_bar=False,
        reset_num_timesteps=False # Continue from where we left off
    )
    
    # 6. Save final model
    os.makedirs(MODEL_SAVE_PATH, exist_ok=True)
    final_path = os.path.join(MODEL_SAVE_PATH, "30mil")
    model.save(final_path)
    logger.info(f"Training complete. Model saved to {final_path}")
    
    # 7. Basic Evaluation
    logger.info("Running evaluation games...")
    obs, info = env.reset()
    total_rewards = 0
    num_eval_games = 10
    games_played = 0
    
    while games_played < num_eval_games:
        action_masks = get_action_masks(env)
        action, _states = model.predict(obs, action_masks=action_masks, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)
        total_rewards += reward
        
        if terminated or truncated:
            games_played += 1
            logger.info(f"Game {games_played} finished. Score: {info['score']}")
            obs, info = env.reset()
            
    logger.info(f"Average Reward (Full): {total_rewards/num_eval_games:.2f}")

if __name__ == "__main__":
    train()
