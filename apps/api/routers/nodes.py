"""Node query routes for graph visualization."""

from __future__ import annotations

from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, Query
from fastapi import status
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from apps.api.dependencies import get_current_user, get_db
from apps.api.models.graph import Graph
from apps.api.models.node import Node
from apps.api.models.user import User
from apps.api.models.workspace import Workspace
from apps.api.schemas.node import NodeRead

router = APIRouter(prefix="/graphs/{graph_id}/nodes", tags=["nodes"])


@router.get("", response_model=list[NodeRead])
async def list_nodes(
    graph_id: UUID,
    node_type: str | None = Query(None),
    cluster_id: str | None = Query(None),
    page: int = Query(1, ge=1),
    size: int = Query(100, ge=1, le=500),
    session: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
) -> list[NodeRead]:
    graph = await session.scalar(select(Graph).where(Graph.id == graph_id))
    if graph is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Graph not found")

    workspace = await session.scalar(
        select(Workspace).where(Workspace.id == graph.workspace_id, Workspace.owner_id == current_user.id)
    )
    if workspace is None:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Forbidden")

    offset = (page - 1) * size
    query = select(Node).where(Node.graph_id == graph.id)
    if node_type:
        query = query.where(Node.node_type == node_type)
    if cluster_id:
        query = query.where(Node.cluster_id == cluster_id)
    query = query.order_by(Node.created_at.asc()).offset(offset).limit(size)
    rows = await session.scalars(query)
    return [
        NodeRead(
            id=node.id,
            graph_id=node.graph_id,
            document_id=node.document_id,
            label=node.label,
            node_type=node.node_type,
            tier=node.tier,
            cluster_id=node.cluster_id,
            metadata=node.metadata_json,
            created_at=node.created_at,
        )
        for node in rows
    ]