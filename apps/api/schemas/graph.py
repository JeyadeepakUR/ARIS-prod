"""Pydantic schemas for graph build routes."""

from __future__ import annotations

from datetime import datetime
from uuid import UUID

from pydantic import BaseModel, Field


class GraphBuildRequest(BaseModel):
    """Payload to request graph construction from ready documents."""

    document_ids: list[UUID] = Field(min_length=2)
    strategy: str = Field(default="sequential", min_length=1, max_length=64)
    keyword: str | None = Field(default=None, max_length=128)
    metadata_field: str | None = Field(default=None, max_length=128)
    metadata_value: str | None = Field(default=None, max_length=256)
    plan_strategy: str = Field(default="weak-evidence", min_length=1, max_length=64)
    max_plan_actions: int = Field(default=10, ge=1, le=100)


class GraphRead(BaseModel):
    """Graph resource representation."""

    id: UUID
    workspace_id: UUID
    status: str
    metadata: dict[str, object]
    created_at: datetime


class GraphBuildResponse(BaseModel):
    """Response after enqueueing graph build."""

    graph: GraphRead
    job_id: UUID
    status: str
    queued: bool


class GraphSummary(BaseModel):
    """Aggregated statistics for a single graph."""

    id: UUID
    workspace_id: UUID
    status: str
    node_count: int
    edge_count: int
    bridge_edge_count: int
    hypothesis_count: int
    created_at: datetime
