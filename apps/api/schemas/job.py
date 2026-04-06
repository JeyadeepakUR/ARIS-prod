"""Pydantic schemas for async jobs."""

from __future__ import annotations

from datetime import datetime
from uuid import UUID

from pydantic import BaseModel


class JobRead(BaseModel):
    """Async job response model."""

    id: UUID
    workspace_id: UUID | None
    job_type: str
    status: str
    payload: dict[str, object]
    result: dict[str, object]
    error: str | None
    created_at: datetime
    started_at: datetime | None
    completed_at: datetime | None
