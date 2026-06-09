from uuid import UUID
from typing import Optional, List

from sqlalchemy import select, update, delete, exists
from sqlalchemy.ext.asyncio import AsyncSession

from src.repository.schemas import GameInstanceORM


class GameRepository:
    def __init__(self, session: AsyncSession):
        self.session = session

    async def create(self, game_code: str, config_type: str, nr_of_players: int, bot_types: List[str], state: dict) -> GameInstanceORM:
        game = GameInstanceORM(
            game_code=game_code,
            config_type=config_type,
            nr_of_players=nr_of_players,
            bot_types=bot_types,
            state=state
        )
        self.session.add(game)
        await self.session.flush()
        return game

    async def get_by_id(self, game_id: UUID) -> Optional[GameInstanceORM]:
        result = await self.session.execute(
            select(GameInstanceORM).where(GameInstanceORM.id == game_id)
        )
        return result.scalar_one_or_none()

    async def get_by_code(self, game_code: str) -> Optional[GameInstanceORM]:
        result = await self.session.execute(
            select(GameInstanceORM)
            .where(GameInstanceORM.game_code == game_code)
            .order_by(GameInstanceORM.created_at.desc())
            .limit(1)
        )
        return result.scalar_one_or_none()

    async def is_code_taken(self, game_code: str) -> bool:
        result = await self.session.execute(
            select(exists().where(GameInstanceORM.game_code == game_code))
        )
        return result.scalar()

    async def update_state(self, game_id: UUID, state: dict) -> None:
        await self.session.execute(
            update(GameInstanceORM)
            .where(GameInstanceORM.id == game_id)
            .values(state=state)
        )

    async def delete(self, game_id: UUID) -> None:
        await self.session.execute(
            delete(GameInstanceORM).where(GameInstanceORM.id == game_id)
        )

    async def delete_by_code(self, game_code: str) -> None:
        await self.session.execute(
            delete(GameInstanceORM).where(GameInstanceORM.game_code == game_code)
        )

    async def delete_all(self) -> None:
        await self.session.execute(delete(GameInstanceORM))

    async def list_all(self) -> List[GameInstanceORM]:
        result = await self.session.execute(
            select(GameInstanceORM).order_by(GameInstanceORM.created_at.desc())
        )
        return list(result.scalars().all())
