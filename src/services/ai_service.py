import asyncio
from typing import Dict, Optional
from src.agents.agent_base import AgentBase
from src.agents.agent_factory import AgentFactory
from src.models.game_config import GameSettings

class AIService:
    """
    Service responsible for managing and caching AI agent instances.
    Ensures that heavy models (like SB3 or Torch-based agents) are only loaded once.
    """
    def __init__(self):
        self._agent_cache: Dict[str, AgentBase] = {}
        self._lock = asyncio.Lock()

    def _get_cache_key(self, agent_type: str, settings: GameSettings) -> str:
        # We use agent_type and the configuration name (full/micro/mini) as the key
        return f"{agent_type}_{settings.name}"

    async def get_agent(self, agent_type: str, settings: GameSettings) -> AgentBase:
        """
        Returns a cached instance of an agent, or creates a new one if not found.
        """
        cache_key = self._get_cache_key(agent_type, settings)

        # Double-checked locking pattern for efficiency
        if cache_key in self._agent_cache:
            return self._agent_cache[cache_key]

        async with self._lock:
            # Re-check inside lock
            if cache_key in self._agent_cache:
                return self._agent_cache[cache_key]

            print(f"[AIService] Instantiating and caching new agent: {cache_key}")
            # Agent creation can be CPU-bound, but since we are in an async method,
            # we just call it. For very heavy initializations, we could use run_in_executor.
            agent = AgentFactory.create_agent(agent_type=agent_type, game_config=settings)
            self._agent_cache[cache_key] = agent
            return agent

# Singleton instance
_ai_service_instance: Optional[AIService] = None

def get_ai_service() -> AIService:
    """
    Dependency provider for AIService. Returns a singleton instance.
    """
    global _ai_service_instance
    if _ai_service_instance is None:
        _ai_service_instance = AIService()
    return _ai_service_instance
