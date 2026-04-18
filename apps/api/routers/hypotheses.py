"""Hypothesis routes for cross-domain bridge investigation."""

from __future__ import annotations

from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, Query
from fastapi import status
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from apps.api.dependencies import get_current_user, get_db
from apps.api.models.graph import Graph
from apps.api.models.hypothesis import Hypothesis
from apps.api.models.user import User
from apps.api.models.workspace import Workspace
from apps.api.schemas.hypothesis import HypothesisRead, HypothesisStatusUpdate

router = APIRouter(prefix="/workspaces/{workspace_id}/graphs/{graph_id}/hypotheses", tags=["hypotheses"])


async def get_workspace_or_403(
    workspace_id: UUID,
    current_user: User,
    session: AsyncSession,
) -> Workspace:
    workspace = await session.scalar(
        select(Workspace).where(Workspace.id == workspace_id, Workspace.owner_id == current_user.id)
    )
    if workspace is None:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Forbidden")
    return workspace


@router.get("", response_model=list[HypothesisRead])
async def list_hypotheses(
    workspace_id: UUID,
    graph_id: UUID,
    hypothesis_status: str | None = Query(None, alias="status"),
    min_confidence: float | None = Query(None, ge=0.0, le=1.0),
    page: int = Query(1, ge=1),
    size: int = Query(50, ge=1, le=200),
    session: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
) -> list[HypothesisRead]:
    workspace = await get_workspace_or_403(workspace_id, current_user, session)
    graph = await session.scalar(select(Graph).where(Graph.id == graph_id, Graph.workspace_id == workspace.id))
    if graph is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Graph not found")

    offset = (page - 1) * size
    query = select(Hypothesis).where(Hypothesis.graph_id == graph.id)
    if hypothesis_status:
        query = query.where(Hypothesis.status == hypothesis_status)
    if min_confidence is not None:
        query = query.where(Hypothesis.confidence >= min_confidence)
    query = query.order_by(Hypothesis.confidence.desc(), Hypothesis.created_at.desc()).offset(offset).limit(size)
    rows = await session.scalars(query)
    return [
        HypothesisRead(
            id=hypothesis.id,
            graph_id=hypothesis.graph_id,
            edge_id=hypothesis.edge_id,
            hypothesis_text=hypothesis.hypothesis_text,
            confidence=hypothesis.confidence,
            status=hypothesis.status,
            created_at=hypothesis.created_at,
        )
        for hypothesis in rows
    ]


@router.patch("/{hypothesis_id}", response_model=HypothesisRead)
async def update_hypothesis_status(
    workspace_id: UUID,
    graph_id: UUID,
    hypothesis_id: UUID,
    payload: HypothesisStatusUpdate,
    session: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
) -> HypothesisRead:
    workspace = await get_workspace_or_403(workspace_id, current_user, session)
    graph = await session.scalar(select(Graph).where(Graph.id == graph_id, Graph.workspace_id == workspace.id))
    if graph is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Graph not found")

    hypothesis = await session.scalar(
        select(Hypothesis).where(Hypothesis.id == hypothesis_id, Hypothesis.graph_id == graph.id)
    )
    if hypothesis is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Hypothesis not found")

    hypothesis.status = payload.status.value
    await session.commit()
    await session.refresh(hypothesis)

    return HypothesisRead(
        id=hypothesis.id,
        graph_id=hypothesis.graph_id,
        edge_id=hypothesis.edge_id,
        hypothesis_text=hypothesis.hypothesis_text,
        confidence=hypothesis.confidence,
        status=hypothesis.status,
        created_at=hypothesis.created_at,
    )
