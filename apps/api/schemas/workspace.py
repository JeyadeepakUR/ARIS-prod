"""Pydantic schemas for workspace endpoints."""

from __future__ import annotations

from datetime import datetime
from uuid import UUID

from pydantic import BaseModel, Field


class WorkspaceCreateRequest(BaseModel):
    """Create workspace payload."""

    name: str = Field(min_length=1, max_length=255)
    slug: str = Field(min_length=1, max_length=255)
    settings: dict[str, object] = Field(default_factory=dict)


class WorkspaceRead(BaseModel):
    """Workspace response model."""

    id: UUID
    owner_id: UUID
    name: str
    slug: str
    settings: dict[str, object]
    created_at: datetime
