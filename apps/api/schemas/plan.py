"""Pydantic schemas for plan generation routes."""

from __future__ import annotations

from datetime import datetime
from uuid import UUID

from pydantic import BaseModel, Field


class PlanGenerateRequest(BaseModel):
    """Payload to generate plan actions from an existing graph."""

    strategy: str = Field(default="weak-evidence", min_length=1, max_length=64)
    max_actions: int = Field(default=10, ge=1, le=100)


class PlanActionRead(BaseModel):
    """Plan action representation."""

    id: UUID
    graph_id: UUID
    action_type: str
    description: str
    evidence: str
    rationale: str
    priority: float
    status: str
    metadata: dict[str, object]
    created_at: datetime


class PlanGenerateResponse(BaseModel):
    """Response after enqueueing plan generation."""

    graph_id: UUID
    job_id: UUID
    status: str
    queued: bool
