"""Async job polling routes."""

from __future__ import annotations

from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from apps.api.dependencies import get_current_user, get_db
from apps.api.models.job import AsyncJob
from apps.api.models.user import User
from apps.api.models.workspace import Workspace
from apps.api.schemas.job import JobRead

router = APIRouter(prefix="/jobs", tags=["jobs"])


@router.get("/{job_id}", response_model=JobRead)
async def get_job(
    job_id: UUID,
    session: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
) -> JobRead:
    job = await session.scalar(select(AsyncJob).where(AsyncJob.id == job_id))
    if job is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Job not found")

    if job.workspace_id is not None:
        workspace = await session.scalar(
            select(Workspace).where(
                Workspace.id == job.workspace_id,
                Workspace.owner_id == current_user.id,
            )
        )
        if workspace is None:
            raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Forbidden")

    return JobRead.model_validate(job, from_attributes=True)
