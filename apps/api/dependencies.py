"""Dependency functions used by FastAPI routes."""

from collections.abc import AsyncGenerator
from uuid import UUID

from fastapi import Depends, HTTPException, status
from fastapi.security import OAuth2PasswordBearer
from jose import JWTError, jwt
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from apps.api.config import Settings, get_settings
from apps.api.core.database import get_db_session
from apps.api.models.user import User


oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/auth/login")


async def get_db() -> AsyncGenerator[AsyncSession, None]:
    """Yield an async database session."""

    async for session in get_db_session():
        yield session


def get_config() -> Settings:
    """Return app settings for dependency injection."""

    return get_settings()


async def get_current_user(
    token: str = Depends(oauth2_scheme),
    session: AsyncSession = Depends(get_db),
    settings: Settings = Depends(get_config),
) -> User:
    """Resolve bearer token to the authenticated user."""

    unauthorized = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Invalid authentication credentials",
    )

    try:
        payload = jwt.decode(token, settings.jwt_public_key, algorithms=[settings.jwt_algorithm])
    except JWTError as exc:
        raise unauthorized from exc

    if payload.get("type") != "access":
        raise unauthorized

    subject = payload.get("sub")
    if not isinstance(subject, str):
        raise unauthorized

    try:
        user_id = UUID(subject)
    except ValueError as exc:
        raise unauthorized from exc

    user = await session.scalar(select(User).where(User.id == user_id))
    if user is None:
        raise unauthorized

    return user
