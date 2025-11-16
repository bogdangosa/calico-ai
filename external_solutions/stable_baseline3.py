from stable_baselines3 import DQN
from stable_baselines3.common.env_checker import check_env

from external_solutions.CalicoGymEnv import CalicoGymEnv

env = CalicoGymEnv()
check_env(env)  # checks Gym compatibility

model = DQN("MlpPolicy", env, verbose=1)
model.learn(total_timesteps=100_000)
