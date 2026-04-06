"""Pydantic schemas for graph hypotheses."""

from __future__ import annotations

from datetime import datetime
from enum import Enum
from uuid import UUID

from pydantic import BaseModel


class HypothesisStatus(str, Enum):
    proposed = "proposed"
    investigating = "investigating"
    accepted = "accepted"
    rejected = "rejected"


class HypothesisRead(BaseModel):
    """Hypothesis response payload."""

    id: UUID
    graph_id: UUID
    edge_id: UUID
    hypothesis_text: str
    confidence: float
    status: HypothesisStatus
    created_at: datetime


class HypothesisStatusUpdate(BaseModel):
    """Patch payload for hypothesis workflow status."""

    status: HypothesisStatus
