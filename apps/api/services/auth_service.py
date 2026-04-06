"""Authentication service handling registration and JWT token flows."""

from __future__ import annotations

import hashlib
from typing import cast
from datetime import UTC, datetime, timedelta
from uuid import UUID

import bcrypt
from jose import JWTError, jwt
from sqlalchemy import delete, select
from sqlalchemy.ext.asyncio import AsyncSession

from apps.api.config import Settings
from apps.api.models.user import RefreshToken, User


class AuthService:
    """Business logic for auth endpoints."""

    def __init__(self, session: AsyncSession, settings: Settings) -> None:
        self._session = session
        self._settings = settings

    async def register(self, email: str, password: str) -> tuple[User, str, str]:
        """Create user and issue initial access/refresh tokens."""

        existing = await self._session.scalar(select(User).where(User.email == email.lower()))
        if existing is not None:
            msg = "Email is already registered"
            raise ValueError(msg)

        user = User(email=email.lower(), hashed_pw=self._hash_password(password))
        self._session.add(user)
        await self._session.flush()

        access_token = self._create_access_token(str(user.id))
        refresh_token = self._create_refresh_token(str(user.id))
        await self._store_refresh_token(user_id=user.id, refresh_token=refresh_token)

        await self._session.commit()
        await self._session.refresh(user)
        return user, access_token, refresh_token

    async def login(self, email: str, password: str) -> tuple[User, str, str]:
        """Verify credentials and issue rotated access/refresh tokens."""

        user = await self._session.scalar(select(User).where(User.email == email.lower()))
        if user is None or not self._verify_password(password, user.hashed_pw):
            msg = "Invalid credentials"
            raise PermissionError(msg)

        access_token = self._create_access_token(str(user.id))
        refresh_token = self._create_refresh_token(str(user.id))
        await self._store_refresh_token(user_id=user.id, refresh_token=refresh_token)
        await self._session.commit()
        return user, access_token, refresh_token

    async def refresh_access_token(self, refresh_token: str) -> str:
        """Rotate refresh token and return a new short-lived access token."""

        payload = self._decode_token(refresh_token)
        if payload.get("type") != "refresh":
            msg = "Invalid refresh token type"
            raise PermissionError(msg)

        user_id = payload.get("sub")
        if not isinstance(user_id, str):
            msg = "Invalid token subject"
            raise PermissionError(msg)

        try:
            parsed_user_id = UUID(user_id)
        except ValueError as exc:
            msg = "Invalid token subject"
            raise PermissionError(msg) from exc

        rows = await self._session.scalars(
            select(RefreshToken).where(RefreshToken.user_id == parsed_user_id)
        )
        matching_token = next(
            (
                row
                for row in rows
                if self._verify_password(self._refresh_token_digest(refresh_token), row.token_hash)
            ),
            None,
        )

        if matching_token is None:
            msg = "Refresh token is invalid or rotated"
            raise PermissionError(msg)

        await self._session.delete(matching_token)
        new_refresh = self._create_refresh_token(user_id)
        await self._store_refresh_token(user_id=parsed_user_id, refresh_token=new_refresh)
        await self._session.commit()
        return self._create_access_token(user_id)

    async def revoke_user_refresh_tokens(self, user_id: str) -> None:
        """Delete all stored refresh tokens for a user."""

        await self._session.execute(delete(RefreshToken).where(RefreshToken.user_id == user_id))
        await self._session.commit()

    def _hash_password(self, password: str) -> str:
        return bcrypt.hashpw(password.encode("utf-8"), bcrypt.gensalt()).decode("utf-8")

    def _verify_password(self, plain_password: str, hashed_password: str) -> bool:
        return bcrypt.checkpw(plain_password.encode("utf-8"), hashed_password.encode("utf-8"))

    def _refresh_token_digest(self, refresh_token: str) -> str:
        return hashlib.sha256(refresh_token.encode("utf-8")).hexdigest()

    def _create_access_token(self, subject: str) -> str:
        expire = datetime.now(UTC) + timedelta(minutes=self._settings.jwt_access_token_expire_minutes)
        payload = {"sub": subject, "type": "access", "exp": expire}
        encoded = jwt.encode(
            claims=payload,
            key=self._settings.jwt_private_key,
            algorithm=self._settings.jwt_algorithm,
        )
        return encoded

    def _create_refresh_token(self, subject: str) -> str:
        expire = datetime.now(UTC) + timedelta(days=self._settings.jwt_refresh_token_expire_days)
        payload = {"sub": subject, "type": "refresh", "exp": expire}
        encoded = jwt.encode(
            claims=payload,
            key=self._settings.jwt_private_key,
            algorithm=self._settings.jwt_algorithm,
        )
        return encoded

    def _decode_token(self, token: str) -> dict[str, object]:
        try:
            decoded = jwt.decode(
                token,
                self._settings.jwt_public_key,
                algorithms=[self._settings.jwt_algorithm],
            )
            return cast(dict[str, object], decoded)
        except JWTError as exc:
            msg = "Invalid token"
            raise PermissionError(msg) from exc

    async def _store_refresh_token(self, user_id: object, refresh_token: str) -> None:
        payload = self._decode_token(refresh_token)
        exp = payload.get("exp")
        if not isinstance(exp, int):
            msg = "Invalid refresh token payload"
            raise PermissionError(msg)

        expires_at = datetime.fromtimestamp(exp, tz=UTC)
        token_entry = RefreshToken(
            user_id=user_id,
            token_hash=self._hash_password(self._refresh_token_digest(refresh_token)),
            expires_at=expires_at,
        )
        self._session.add(token_entry)
