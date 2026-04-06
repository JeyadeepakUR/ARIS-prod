"""Node query routes for graph visualization."""

from __future__ import annotations

from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, status
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

    rows = await session.scalars(select(Node).where(Node.graph_id == graph.id).order_by(Node.created_at.asc()))
    return [
        NodeRead(
            id=node.id,
            graph_id=node.graph_id,
            document_id=node.document_id,
            label=node.label,
            node_type=node.node_type,
            metadata=node.metadata_json,
            created_at=node.created_at,
        )
        for node in rows
    ]