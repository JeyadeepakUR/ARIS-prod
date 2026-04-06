"""Authentication API routes."""

from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.ext.asyncio import AsyncSession

from apps.api.config import Settings
from apps.api.dependencies import get_config, get_db
from apps.api.schemas.auth import (
    AccessTokenResponse,
    LoginRequest,
    RefreshRequest,
    RegisterRequest,
    TokenResponse,
    UserRead,
)
from apps.api.services.auth_service import AuthService

router = APIRouter(prefix="/auth", tags=["auth"])


@router.post("/register", response_model=TokenResponse, status_code=status.HTTP_201_CREATED)
async def register(
    payload: RegisterRequest,
    session: AsyncSession = Depends(get_db),
    settings: Settings = Depends(get_config),
) -> TokenResponse:
    service = AuthService(session=session, settings=settings)
    try:
        user, access_token, refresh_token = await service.register(payload.email, payload.password)
    except ValueError as exc:
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail=str(exc)) from exc

    return TokenResponse(
        access_token=access_token,
        refresh_token=refresh_token,
        user=UserRead.model_validate(user, from_attributes=True),
    )


@router.post("/login", response_model=TokenResponse)
async def login(
    payload: LoginRequest,
    session: AsyncSession = Depends(get_db),
    settings: Settings = Depends(get_config),
) -> TokenResponse:
    service = AuthService(session=session, settings=settings)
    try:
        user, access_token, refresh_token = await service.login(payload.email, payload.password)
    except PermissionError as exc:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail=str(exc)) from exc

    return TokenResponse(
        access_token=access_token,
        refresh_token=refresh_token,
        user=UserRead.model_validate(user, from_attributes=True),
    )


@router.post("/refresh", response_model=AccessTokenResponse)
async def refresh(
    payload: RefreshRequest,
    session: AsyncSession = Depends(get_db),
    settings: Settings = Depends(get_config),
) -> AccessTokenResponse:
    service = AuthService(session=session, settings=settings)
    try:
        access_token = await service.refresh_access_token(payload.refresh_token)
    except PermissionError as exc:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail=str(exc)) from exc

    return AccessTokenResponse(access_token=access_token)
