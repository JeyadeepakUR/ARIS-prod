"""Pydantic schemas for chunk evidence look-ups.

Chunk evidence powers the UI's traceability story: any concept node, bridge
edge, contradiction, or research gap stores chunk IDs in its metadata, and the
frontend resolves those IDs back into the actual paragraph text + page +
section + parent paper title via the chunks endpoint.
"""

from __future__ import annotations

from uuid import UUID

from pydantic import BaseModel


class ChunkEvidence(BaseModel):
    """Single chunk of source-paper text used as evidence."""

    chunk_id: UUID
    document_id: UUID
    document_title: str
    document_filename: str
    document_s3_key: str | None = None
    page_number: int | None = None
    section: str | None = None
    chunk_index: int
    content: str
