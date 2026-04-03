from src.engine.environments.calico_env import CalicoEnv
from src.engine.scoring.scoring import ScoringCalculator
from src.ui.console_ui import CalicoConsoleUI
from src.ui.table_renderer import TableRenderer
from src.utils.config import load_config

config = load_config("../../config/calico_settings.json")
env = CalicoEnv(config)
scorer = ScoringCalculator(config)
renderer = TableRenderer(config)
ui = CalicoConsoleUI(env,renderer, scorer)

ui.run()