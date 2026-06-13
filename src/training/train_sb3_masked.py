import os
from datetime import datetime
from sb3_contrib import MaskablePPO
from sb3_contrib.common.maskable.utils import get_action_masks
from sb3_contrib.common.wrappers import ActionMasker
from stable_baselines3.common.callbacks import CheckpointCallback, BaseCallback
from loguru import logger

from src.engine.environments.calico_gym_wrapper import CalicoGymWrapper
from src.utils.config import load_config

CONFIG_PATH = "../../config/micro_calico_settings_v2.json"
TOTAL_TIMESTEPS = 10000000
MODEL_SAVE_PATH = "../../agent_models/micro_calico_v2/sb3_masked_ppo/sb3_calico_ppo_10mil_v1.0.zip"
LOAD_MODEL_PATH = None
LOG_DIR = f"../../outputs/logs/sb3_training_micro/{datetime.now().strftime('%Y%m%d-%H%M%S')}"

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
            config = load_config("config/micro_calico_settings_v2.json")
    else:
        config = load_config(CONFIG_PATH)
        
    raw_env = CalicoGymWrapper(config)
    env = ActionMasker(raw_env, mask_fn)
    
    # 3. Load or Initialize the MaskablePPO model
    if LOAD_MODEL_PATH and os.path.exists(LOAD_MODEL_PATH):
        logger.info(f"Loading existing model from {LOAD_MODEL_PATH}...")
        model = MaskablePPO.load(
            LOAD_MODEL_PATH, 
            env=env, 
            tensorboard_log=LOG_DIR,
        )
        reset_num_timesteps = False
    else:
        logger.info("Initializing new MaskablePPO model from scratch...")
        model = MaskablePPO(
            "MultiInputPolicy",
            env,
            verbose=1,
            tensorboard_log=LOG_DIR,
            gamma=0.99,
            learning_rate=3e-4,
        )
        reset_num_timesteps = True
    
    # 4. Setup callbacks
    checkpoint_dir = os.path.dirname(MODEL_SAVE_PATH)
    os.makedirs(checkpoint_dir, exist_ok=True)
    checkpoint_callback = CheckpointCallback(
        save_freq=1000000, # Save every 1M steps
        save_path=checkpoint_dir,
        name_prefix="sb3_full_calico_ppo_checkpoint"
    )

    score_logger = ActualScoreCallback()

    # 5. Train the model
    logger.info(f"Starting SB3 Full Calico training for {TOTAL_TIMESTEPS} timesteps...")
    model.learn(
        total_timesteps=TOTAL_TIMESTEPS,
        callback=[checkpoint_callback, score_logger],
        progress_bar=False,
        reset_num_timesteps=reset_num_timesteps
    )
    
    # 6. Save final model
    model.save(MODEL_SAVE_PATH)
    logger.info(f"Training complete. Model saved to {MODEL_SAVE_PATH}")
    
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
