from src.engine.environments.calico_env import CalicoEnv
from src.ui.console_ui import CalicoConsoleUI
from src.utils.config import load_config

config = load_config("../../config/calico_settings.json")
env = CalicoEnv(config)
ui = CalicoConsoleUI(env)

ui.run()