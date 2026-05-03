"""AgentStreamEvent — per-step telemetry rows emitted by LangGraph agent nodes."""

from __future__ import annotations

import uuid
from datetime import datetime

from sqlalchemy import DateTime, ForeignKey, JSON, String, Text, Uuid, func
from sqlalchemy.orm import Mapped, mapped_column

from apps.api.core.database import Base


class AgentStreamEvent(Base):
    """Single event row emitted by an agent node during a graph build run.

    Rows are short-lived: they are consumed by the SSE endpoint while the
    build is in-flight and deleted once the graph reaches a terminal status.
    """

    __tablename__ = "agent_stream_events"

    id: Mapped[uuid.UUID] = mapped_column(Uuid(as_uuid=True), primary_key=True, default=uuid.uuid4)
    graph_id: Mapped[uuid.UUID] = mapped_column(
        Uuid(as_uuid=True),
        ForeignKey("graphs.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    thread_id: Mapped[str] = mapped_column(String(100), nullable=False, index=True)
    agent_node: Mapped[str] = mapped_column(String(100), nullable=False)
    event_type: Mapped[str] = mapped_column(String(100), nullable=False)
    content: Mapped[dict] = mapped_column(JSON, nullable=False, default=dict)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now(), index=True
    )
