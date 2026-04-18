"""Edge query routes for Sprint 3 graph outputs."""

from __future__ import annotations

from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, Query
from fastapi import status
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from apps.api.dependencies import get_current_user, get_db
from apps.api.models.edge import Edge
from apps.api.models.graph import Graph
from apps.api.models.user import User
from apps.api.models.workspace import Workspace
from apps.api.schemas.edge import EdgeRead

router = APIRouter(prefix="/graphs/{graph_id}/edges", tags=["edges"])


@router.get("", response_model=list[EdgeRead])
async def list_edges(
    graph_id: UUID,
    edge_type: str | None = Query(None),
    edge_category: str | None = Query(None),
    min_confidence: float | None = Query(None, ge=0.0, le=1.0),
    page: int = Query(1, ge=1),
    size: int = Query(100, ge=1, le=500),
    session: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
) -> list[EdgeRead]:
    graph = await session.scalar(select(Graph).where(Graph.id == graph_id))
    if graph is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Graph not found")

    workspace = await session.scalar(
        select(Workspace).where(Workspace.id == graph.workspace_id, Workspace.owner_id == current_user.id)
    )
    if workspace is None:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Forbidden")

    offset = (page - 1) * size
    from sqlalchemy import Float
    query = select(Edge).where(Edge.graph_id == graph.id)
    if edge_type:
        query = query.where(Edge.edge_type == edge_type)
    if edge_category:
        query = query.where(Edge.edge_category == edge_category)
    if min_confidence is not None:
        query = query.where(Edge.confidence >= min_confidence)
    query = query.order_by(Edge.confidence.desc(), Edge.created_at.asc()).offset(offset).limit(size)
    rows = await session.scalars(query)
    return [
        EdgeRead(
            id=edge.id,
            graph_id=edge.graph_id,
            source_node_id=edge.source_node_id,
            target_node_id=edge.target_node_id,
            edge_type=edge.edge_type,
            confidence=edge.confidence,
            evidence={
                "text": edge.evidence,
                "reasoning_trace_id": str(edge.reasoning_trace_id),
                "confidence": edge.confidence,
            },
            metadata=edge.metadata_json,
            created_at=edge.created_at,
        )
        for edge in rows
    ]
