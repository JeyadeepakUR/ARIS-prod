"""Contradiction model — claim pairs across authors with LLM reasoning."""

from __future__ import annotations

import uuid
from datetime import datetime

from sqlalchemy import DateTime, Float, ForeignKey, String, Text, Uuid, func
from sqlalchemy.orm import Mapped, mapped_column

from apps.api.core.database import Base


class Contradiction(Base):
    __tablename__ = "contradictions"

    id: Mapped[uuid.UUID] = mapped_column(Uuid(as_uuid=True), primary_key=True, default=uuid.uuid4)
    graph_id: Mapped[uuid.UUID] = mapped_column(
        Uuid(as_uuid=True), ForeignKey("graphs.id", ondelete="CASCADE"),
        nullable=False, index=True
    )
    claim_a_edge_id: Mapped[uuid.UUID | None] = mapped_column(
        Uuid(as_uuid=True), nullable=True
    )
    claim_b_edge_id: Mapped[uuid.UUID | None] = mapped_column(
        Uuid(as_uuid=True), nullable=True
    )
    claim_a_chunk_id: Mapped[uuid.UUID | None] = mapped_column(
        Uuid(as_uuid=True), ForeignKey("document_chunks.id", ondelete="SET NULL"), nullable=True
    )
    claim_b_chunk_id: Mapped[uuid.UUID | None] = mapped_column(
        Uuid(as_uuid=True), ForeignKey("document_chunks.id", ondelete="SET NULL"), nullable=True
    )
    author_a: Mapped[str | None] = mapped_column(String(500), nullable=True)
    author_b: Mapped[str | None] = mapped_column(String(500), nullable=True)
    claim_a_text: Mapped[str] = mapped_column(Text, nullable=False)
    claim_b_text: Mapped[str] = mapped_column(Text, nullable=False)
    contradiction_type: Mapped[str] = mapped_column(
        String(50), nullable=False, default="direct"
    )
    severity: Mapped[float] = mapped_column(Float, nullable=False, default=0.5)
    llm_reasoning: Mapped[str | None] = mapped_column(Text, nullable=True)
    status: Mapped[str] = mapped_column(
        String(20), nullable=False, default="open", server_default="open"
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now()
    )
