from uuid import UUID
from typing import List, Optional
from sqlalchemy import select, delete
from sqlalchemy.ext.asyncio import AsyncSession
from src.repository.schemas import PlayerORM
from src.api.models import PlayerCreate

class PlayerRepository:
    def __init__(self, session: AsyncSession):
        self.session = session

    async def add_player(self, game_id: UUID, player_data: PlayerCreate) -> PlayerORM:
        player = PlayerORM(
            game_id=game_id,
            player_name=player_data.player_name,
            player_type=player_data.player_type,
            order_index=player_data.order_index
        )
        self.session.add(player)
        await self.session.flush()
        return player

    async def get_players_by_game(self, game_id: UUID) -> List[PlayerORM]:
        result = await self.session.execute(
            select(PlayerORM)
            .where(PlayerORM.game_id == game_id)
            .order_by(PlayerORM.order_index)
        )
        return list(result.scalars().all())

    async def delete_player(self, player_id: UUID) -> None:
        await self.session.execute(
            delete(PlayerORM).where(PlayerORM.id == player_id)
        )

    async def get_player(self, player_id: UUID) -> Optional[PlayerORM]:
        result = await self.session.execute(
            select(PlayerORM).where(PlayerORM.id == player_id)
        )
        return result.scalar_one_or_none()
