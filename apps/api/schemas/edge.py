"""Pydantic schemas for graph edges."""

from __future__ import annotations

from datetime import datetime
from uuid import UUID

from pydantic import BaseModel


class EdgeRead(BaseModel):
    """Edge response model with evidence payload."""

    id: UUID
    graph_id: UUID
    source_node_id: UUID
    target_node_id: UUID
    edge_type: str
    edge_category: str = "INTRA_DOMAIN"
    bridge_concept: str | None = None
    confidence: float
    evidence: dict[str, object]
    metadata: dict[str, object]
    created_at: datetime


class EdgeDetail(EdgeRead):
    """Detailed edge view, currently aligned with EdgeRead payload."""
