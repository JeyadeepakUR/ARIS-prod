"""Pydantic schemas for graph nodes."""

from __future__ import annotations

from datetime import datetime
from uuid import UUID

from pydantic import BaseModel


class NodeRead(BaseModel):
    """Graph node representation."""

    id: UUID
    graph_id: UUID
    document_id: UUID | None
    label: str
    node_type: str
    tier: int = 3
    cluster_id: str | None = None
    metadata: dict[str, object]
    created_at: datetime