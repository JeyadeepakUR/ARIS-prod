"""Hypothesis model persisted for cross-domain bridge edges."""

from __future__ import annotations

import uuid
from datetime import datetime

from sqlalchemy import DateTime, Float, ForeignKey, String, Text, Uuid, func
from sqlalchemy.orm import Mapped, mapped_column, relationship

from apps.api.core.database import Base


class Hypothesis(Base):
    """Generated research hypothesis for an inter-domain bridge edge."""

    __tablename__ = "hypotheses"

    id: Mapped[uuid.UUID] = mapped_column(Uuid(as_uuid=True), primary_key=True, default=uuid.uuid4)
    graph_id: Mapped[uuid.UUID] = mapped_column(
        Uuid(as_uuid=True), ForeignKey("graphs.id", ondelete="CASCADE"), nullable=False, index=True
    )
    edge_id: Mapped[uuid.UUID] = mapped_column(
        Uuid(as_uuid=True), ForeignKey("edges.id", ondelete="CASCADE"), nullable=False, index=True
    )
    hypothesis_text: Mapped[str] = mapped_column(Text, nullable=False)
    confidence: Mapped[float] = mapped_column(Float, nullable=False)
    status: Mapped[str] = mapped_column(String(50), nullable=False, default="proposed")
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now()
    )

    graph = relationship("Graph", lazy="joined")
    edge = relationship("Edge", lazy="joined")
