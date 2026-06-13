import json
from pathlib import Path

from loguru import logger

from src.models.game_config import GameSettings


from functools import lru_cache

@lru_cache(maxsize=16)
def load_config(config_path: str) -> GameSettings:
    """
    Reads a JSON file and parses it into a GameSettings object.
    """
    path = Path(config_path)

    if not path.exists():
        logger.error(f"Config file not found at: {path.absolute()}")
        raise FileNotFoundError(f"Could not find {config_path}")

    try:
        with open(path, "r") as f:
            data = json.load(f)

        config = GameSettings(**data)

        logger.info(f"Configuration loaded successfully from {path.name}")
        return config

    except Exception as e:
        logger.error(f"Failed to parse config: {e}")
        raise