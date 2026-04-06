"""Async SQLAlchemy engine and session management."""

from collections.abc import AsyncGenerator

from sqlalchemy.ext.asyncio import (
    AsyncEngine,
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)
from sqlalchemy.orm import DeclarativeBase

from apps.api.config import Settings


class Base(DeclarativeBase):
    """Base class for all SQLAlchemy models."""


engine: AsyncEngine | None = None
SessionLocal: async_sessionmaker[AsyncSession] | None = None


def init_database(settings: Settings) -> None:
    """Initialize async engine and session factory from application settings."""

    global engine, SessionLocal
    engine = create_async_engine(settings.database_url, echo=settings.database_echo, future=True)
    SessionLocal = async_sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)


def get_engine() -> AsyncEngine | None:
    """Return the initialized SQLAlchemy engine instance."""

    return engine


async def dispose_database() -> None:
    """Dispose the async SQLAlchemy engine."""

    if engine is not None:
        await engine.dispose()


async def get_db_session() -> AsyncGenerator[AsyncSession, None]:
    """Yield an async DB session for FastAPI dependencies."""

    if SessionLocal is None:
        msg = "Database is not initialized. Call init_database() first."
        raise RuntimeError(msg)

    async with SessionLocal() as session:
        yield session
