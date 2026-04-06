"""Workspace CRUD routes for Sprint 2."""

from __future__ import annotations

from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from apps.api.dependencies import get_current_user, get_db
from apps.api.models.user import User
from apps.api.models.workspace import Workspace
from apps.api.schemas.workspace import WorkspaceCreateRequest, WorkspaceRead

router = APIRouter(prefix="/workspaces", tags=["workspaces"])


@router.post("", response_model=WorkspaceRead, status_code=status.HTTP_201_CREATED)
async def create_workspace(
    payload: WorkspaceCreateRequest,
    session: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
) -> WorkspaceRead:
    existing = await session.scalar(
        select(Workspace).where(Workspace.slug == payload.slug, Workspace.owner_id == current_user.id)
    )
    if existing is not None:
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail="Workspace slug already exists")

    workspace = Workspace(
        owner_id=current_user.id,
        name=payload.name,
        slug=payload.slug,
        settings=payload.settings,
    )
    session.add(workspace)
    await session.commit()
    await session.refresh(workspace)
    return WorkspaceRead.model_validate(workspace, from_attributes=True)


@router.get("", response_model=list[WorkspaceRead])
async def list_workspaces(
    session: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
) -> list[WorkspaceRead]:
    rows = await session.scalars(
        select(Workspace).where(Workspace.owner_id == current_user.id).order_by(Workspace.created_at.desc())
    )
    return [WorkspaceRead.model_validate(row, from_attributes=True) for row in rows]


@router.get("/{workspace_id}", response_model=WorkspaceRead)
async def get_workspace(
    workspace_id: UUID,
    session: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
) -> WorkspaceRead:
    workspace = await session.scalar(
        select(Workspace).where(Workspace.id == workspace_id, Workspace.owner_id == current_user.id)
    )
    if workspace is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Workspace not found")

    return WorkspaceRead.model_validate(workspace, from_attributes=True)
