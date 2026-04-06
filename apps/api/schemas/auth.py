"""Pydantic models for authentication endpoints."""

from __future__ import annotations

from datetime import datetime
from uuid import UUID

from pydantic import BaseModel, EmailStr, Field


class RegisterRequest(BaseModel):
    """Request payload for user registration."""

    email: EmailStr
    password: str = Field(min_length=8, max_length=128)


class LoginRequest(BaseModel):
    """Request payload for user login."""

    email: EmailStr
    password: str = Field(min_length=8, max_length=128)


class RefreshRequest(BaseModel):
    """Request payload for access token refresh."""

    refresh_token: str = Field(min_length=16)


class UserRead(BaseModel):
    """User response model."""

    id: UUID
    email: EmailStr
    plan: str
    is_verified: bool
    created_at: datetime


class TokenResponse(BaseModel):
    """Response payload for token issuing endpoints."""

    access_token: str
    refresh_token: str
    token_type: str = "bearer"
    user: UserRead


class AccessTokenResponse(BaseModel):
    """Response payload for refresh endpoint."""

    access_token: str
    token_type: str = "bearer"
