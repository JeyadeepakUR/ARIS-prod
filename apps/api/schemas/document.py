"""Pydantic schemas for document ingest endpoints."""

from __future__ import annotations

from datetime import datetime
from uuid import UUID

from pydantic import BaseModel, Field


class DocumentUploadInitRequest(BaseModel):
    """Payload used to request a pre-signed upload URL."""

    filename: str = Field(min_length=1, max_length=500)
    file_format: str = Field(min_length=1, max_length=50)
    file_size_bytes: int | None = Field(default=None, ge=1)


class DocumentRead(BaseModel):
    """Document response model."""

    id: UUID
    workspace_id: UUID
    filename: str
    s3_key: str
    file_format: str
    file_size_bytes: int | None
    status: str
    error_msg: str | None
    metadata: dict[str, object]
    created_at: datetime


class DocumentUploadInitResponse(BaseModel):
    """Response for upload initialization."""

    document: DocumentRead
    job_id: UUID
    status: str
    upload_url: str
    queued: bool
