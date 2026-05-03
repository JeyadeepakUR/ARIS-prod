"""Contradiction routes — list claim conflicts detected during graph build."""

from __future__ import annotations

from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, Query
from fastapi import status as http_status
from pydantic import BaseModel
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from apps.api.dependencies import get_current_user, get_db
from apps.api.models.contradiction import Contradiction
from apps.api.models.graph import Graph
from apps.api.models.user import User
from apps.api.models.workspace import Workspace

router = APIRouter(prefix="/graphs/{graph_id}/contradictions", tags=["contradictions"])


class ContradictionRead(BaseModel):
    id: UUID
    graph_id: UUID
    claim_a_text: str
    claim_b_text: str
    author_a: str | None
    author_b: str | None
    contradiction_type: str
    severity: float
    llm_reasoning: str | None
    status: str
    # Chunk references back to the source paragraphs that produced each claim.
    # Optional because older rows from before traceability work may be null.
    claim_a_chunk_id: UUID | None = None
    claim_b_chunk_id: UUID | None = None
    created_at: str

    model_config = {"from_attributes": True}


@router.get("", response_model=list[ContradictionRead])
async def list_contradictions(
    graph_id: UUID,
    page: int = Query(1, ge=1),
    size: int = Query(50, ge=1, le=200),
    session: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
) -> list[ContradictionRead]:
    graph = await session.scalar(select(Graph).where(Graph.id == graph_id))
    if graph is None:
        raise HTTPException(status_code=http_status.HTTP_404_NOT_FOUND, detail="Graph not found")

    workspace = await session.scalar(
        select(Workspace).where(
            Workspace.id == graph.workspace_id,
            Workspace.owner_id == current_user.id,
        )
    )
    if workspace is None:
        raise HTTPException(status_code=http_status.HTTP_403_FORBIDDEN, detail="Forbidden")

    offset = (page - 1) * size
    rows = await session.scalars(
        select(Contradiction)
        .where(Contradiction.graph_id == graph_id)
        .order_by(Contradiction.severity.desc(), Contradiction.created_at.desc())
        .offset(offset)
        .limit(size)
    )
    return [
        ContradictionRead(
            id=c.id,
            graph_id=c.graph_id,
            claim_a_text=c.claim_a_text,
            claim_b_text=c.claim_b_text,
            author_a=c.author_a,
            author_b=c.author_b,
            contradiction_type=c.contradiction_type,
            severity=c.severity,
            llm_reasoning=c.llm_reasoning,
            status=c.status,
            claim_a_chunk_id=c.claim_a_chunk_id,
            claim_b_chunk_id=c.claim_b_chunk_id,
            created_at=c.created_at.isoformat(),
        )
        for c in rows
    ]
