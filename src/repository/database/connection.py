import asyncio
from collections.abc import AsyncGenerator
from typing import Any

from sqlalchemy import make_url
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from src.utils.settings import settings, fetch_from_env

# Use default values if not provided in settings
POOL_SIZE = getattr(settings.database, "pool_size", 10)
WORKER_COUNT = getattr(settings.database, "worker_count", 1)
POOL_PER_WORKER = getattr(settings.database, "pool_per_worker", 5)

FINAL_POOL_SIZE = max(
    POOL_SIZE,
    WORKER_COUNT * POOL_PER_WORKER,
)
MAX_OVERFLOW = getattr(settings.database, "max_overflow", int(FINAL_POOL_SIZE * 0.5))


def _create_engine():
    """Create a fresh async engine instance."""
    url = settings.get("DATABASE.URL") or settings.get("database", {}).get("url")
    return create_async_engine(
        url,
        pool_pre_ping=True,
        pool_size=FINAL_POOL_SIZE,
        max_overflow=MAX_OVERFLOW,
    )


# Default engine for FastAPI (long-lived process with stable event loop)
engine = _create_engine()

AsyncSessionLocal = async_sessionmaker(
    bind=engine, expire_on_commit=False, class_=AsyncSession
)


async def get_session() -> AsyncGenerator[AsyncSession, Any]:
    """Get a session for FastAPI requests. Commits on success, rolls back on error."""
    async with AsyncSessionLocal() as session:
        try:
            yield session
            await session.commit()
        except Exception:
            await session.rollback()
            raise


async def dispose_engine() -> None:
    """Dispose the engine's connection pool."""
    await engine.dispose()
