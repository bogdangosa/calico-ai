from datetime import datetime
from uuid import UUID, uuid4
from typing import List

from sqlalchemy import JSON, DateTime, String, Integer, ForeignKey
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship


class Base(DeclarativeBase):
    pass


class GameInstanceORM(Base):
    __tablename__ = "game_instances"

    id: Mapped[UUID] = mapped_column(primary_key=True, default=uuid4)
    game_code: Mapped[str] = mapped_column(String(4), unique=True, index=True)
    config_type: Mapped[str] = mapped_column(String(50))
    nr_of_players: Mapped[int] = mapped_column(Integer)
    bot_types: Mapped[list[str]] = mapped_column(JSON)
    state: Mapped[dict] = mapped_column(JSON)
    
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime, default=datetime.utcnow, onupdate=datetime.utcnow
    )

    # Relationship to players
    players: Mapped[List["PlayerORM"]] = relationship(back_populates="game", cascade="all, delete-orphan")


class PlayerORM(Base):
    __tablename__ = "players"

    id: Mapped[UUID] = mapped_column(primary_key=True, default=uuid4)
    game_id: Mapped[UUID] = mapped_column(ForeignKey("game_instances.id"))
    player_type: Mapped[str] = mapped_column(String(20))  # "ai" or "person"
    order_index: Mapped[int] = mapped_column(Integer)
    player_name: Mapped[str] = mapped_column(String(100))
    
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)

    # Relationship back to game
    game: Mapped["GameInstanceORM"] = relationship(back_populates="players")
