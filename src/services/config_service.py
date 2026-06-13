from typing import Dict, Optional
from src.models.game_config import GameSettings
from src.utils.config import load_config
from loguru import logger
import os

class ConfigService:
    """
    Service responsible for managing and caching GameSettings.
    Avoids repeated disk I/O and parsing of configuration files.
    """
    def __init__(self):
        self._cache: Dict[str, GameSettings] = {}

    def get_config(self, config_path: str) -> GameSettings:
        """
        Returns a cached GameSettings object or loads it from disk if not cached.
        """
        # Normalize path to ensure consistent keys
        abs_path = os.path.abspath(config_path)
        
        if abs_path in self._cache:
            logger.debug(f"[ConfigService] Cache hit for: {config_path}")
            return self._cache[abs_path]
        
        logger.info(f"[ConfigService] Cache miss for: {config_path}. Loading from disk...")
        config = load_config(config_path)
        self._cache[abs_path] = config
        return config

    def clear_cache(self):
        """Clears all cached configurations."""
        self._cache.clear()
        logger.info("[ConfigService] Cache cleared.")

# Singleton instance
_config_service_instance: Optional[ConfigService] = None

def get_config_service() -> ConfigService:
    """
    Dependency provider for ConfigService. Returns a singleton instance.
    """
    global _config_service_instance
    if _config_service_instance is None:
        _config_service_instance = ConfigService()
    return _config_service_instance
