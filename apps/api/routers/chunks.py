"""Chunk evidence endpoint.

Resolves chunk IDs (referenced by concept nodes, bridge edges, contradictions,
and gaps) into the actual paragraph text + page + section + parent paper
title that produced them. This is the foundation of the UI's traceability:
"show me the literal sentence in the literal paper that the LLM read".
"""

from __future__ import annotations

from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, Query, status
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from apps.api.dependencies import get_current_user, get_db
from apps.api.models.chunk import DocumentChunk
from apps.api.models.document import Document
from apps.api.models.graph import Graph
from apps.api.models.user import User
from apps.api.models.workspace import Workspace
from apps.api.schemas.chunk import ChunkEvidence

router = APIRouter(prefix="/graphs/{graph_id}/chunks", tags=["chunks"])


def _document_title(doc: Document) -> str:
    """Best-effort human title: doc.metadata.title -> filename -> id prefix."""
    meta = doc.metadata_json or {}
    title = str(meta.get("title") or "").strip()
    if title:
        return title
    if doc.filename:
        # Strip extension for cleaner display.
        return doc.filename.rsplit(".", 1)[0]
    return str(doc.id)[:8]


@router.get("", response_model=list[ChunkEvidence])
async def get_chunk_evidence(
    graph_id: UUID,
    ids: str = Query(..., description="Comma-separated chunk UUIDs"),
    session: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
) -> list[ChunkEvidence]:
    """Return chunk evidence for the given chunk IDs, scoped to the graph's workspace."""

    # Authorise: graph must exist and belong to a workspace owned by the user.
    graph = await session.scalar(select(Graph).where(Graph.id == graph_id))
    if graph is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Graph not found")
    workspace = await session.scalar(
        select(Workspace).where(
            Workspace.id == graph.workspace_id,
            Workspace.owner_id == current_user.id,
        )
    )
    if workspace is None:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Forbidden")

    raw_ids = [token.strip() for token in (ids or "").split(",") if token.strip()]
    chunk_ids: list[UUID] = []
    for token in raw_ids:
        try:
            chunk_ids.append(UUID(token))
        except ValueError:
            # Silently drop garbage IDs rather than 400-ing the whole request.
            continue
    if not chunk_ids:
        return []

    chunk_rows = (await session.scalars(
        select(DocumentChunk).where(DocumentChunk.id.in_(chunk_ids))
    )).all()
    if not chunk_rows:
        return []

    doc_ids = {c.document_id for c in chunk_rows}
    doc_rows = (await session.scalars(
        select(Document).where(
            Document.id.in_(doc_ids),
            Document.workspace_id == graph.workspace_id,
        )
    )).all()
    docs_by_id = {d.id: d for d in doc_rows}

    out: list[ChunkEvidence] = []
    # Per-document dedup of near-identical evidence prefixes — necessary so the
    # UI doesn't show two visually-identical IEEE-header chunks side by side
    # for the same concept (already-extracted graphs may have duplicate
    # source_chunk_ids stored).
    seen_prefix: dict[UUID, set[str]] = {}
    chunk_by_id = {c.id: c for c in chunk_rows}
    # Preserve the user-supplied ID order so the UI gets stable rendering.
    for cid in chunk_ids:
        chunk = chunk_by_id.get(cid)
        if chunk is None:
            continue
        doc = docs_by_id.get(chunk.document_id)
        if doc is None:
            continue
        # Dedup signal: lowercased, whitespace-collapsed first 120 chars.
        content = (chunk.content or "").strip()
        prefix = " ".join(content.lower().split())[:120]
        bucket = seen_prefix.setdefault(doc.id, set())
        if prefix and prefix in bucket:
            continue
        if prefix:
            bucket.add(prefix)
        out.append(
            ChunkEvidence(
                chunk_id=chunk.id,
                document_id=doc.id,
                document_title=_document_title(doc),
                document_filename=doc.filename,
                document_s3_key=doc.s3_key,
                page_number=chunk.page_number,
                section=chunk.section,
                chunk_index=chunk.chunk_index,
                content=chunk.content,
            )
        )
    return out
