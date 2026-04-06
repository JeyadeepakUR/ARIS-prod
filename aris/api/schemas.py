"""Pydantic schemas for ARIS API endpoints."""

from __future__ import annotations

from pydantic import BaseModel, Field


class HealthResponse(BaseModel):
    status: str = Field(default="ok")
    service: str = Field(default="aris")
    version: str


class DocumentInput(BaseModel):
    document_id: str
    domain: str
    text: str


class AnalyzeRequest(BaseModel):
    documents: list[DocumentInput] = Field(min_length=2)
    top_k_bridges: int = Field(default=15, ge=1, le=100)
    max_hypotheses: int = Field(default=20, ge=1, le=100)


class AnalyzeResponse(BaseModel):
    profile_count: int
    contradiction_count: int
    bridge_count: int
    hypothesis_count: int
    contradictions: list[dict[str, object]]
    bridges: list[dict[str, object]]
    hypotheses: list[dict[str, object]]
