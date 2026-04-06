"""Edge model persisted per graph."""

from __future__ import annotations

import uuid
from datetime import datetime

from sqlalchemy import DateTime, Float, ForeignKey, JSON, String, Text, Uuid, func
from sqlalchemy.orm import Mapped, mapped_column

from apps.api.core.database import Base


class Edge(Base):
    """Persisted directed relationship with evidence."""

    __tablename__ = "edges"

    id: Mapped[uuid.UUID] = mapped_column(Uuid(as_uuid=True), primary_key=True)
    graph_id: Mapped[uuid.UUID] = mapped_column(
        Uuid(as_uuid=True), ForeignKey("graphs.id", ondelete="CASCADE"), nullable=False, index=True
    )
    source_node_id: Mapped[uuid.UUID] = mapped_column(
        Uuid(as_uuid=True), ForeignKey("nodes.id", ondelete="CASCADE"), nullable=False, index=True
    )
    target_node_id: Mapped[uuid.UUID] = mapped_column(
        Uuid(as_uuid=True), ForeignKey("nodes.id", ondelete="CASCADE"), nullable=False, index=True
    )
    edge_type: Mapped[str] = mapped_column(String(100), nullable=False)
    edge_category: Mapped[str] = mapped_column(
        String(50), nullable=False, default="INTRA_DOMAIN", server_default="INTRA_DOMAIN", index=True
    )
    bridge_concept: Mapped[str | None] = mapped_column(String(255), nullable=True)
    evidence: Mapped[str] = mapped_column(Text, nullable=False)
    reasoning_trace_id: Mapped[uuid.UUID] = mapped_column(Uuid(as_uuid=True), nullable=False, index=True)
    confidence: Mapped[float] = mapped_column(Float, nullable=False)
    metadata_json: Mapped[dict[str, object]] = mapped_column("metadata", JSON, nullable=False, default=dict)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now()
    )
