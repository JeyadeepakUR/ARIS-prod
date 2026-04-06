"""Graph build routes for Sprint 3."""

from __future__ import annotations

from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from apps.api.config import Settings
from apps.api.dependencies import get_config, get_current_user, get_db
from apps.api.models.document import Document
from apps.api.models.graph import Graph
from apps.api.models.job import AsyncJob
from apps.api.models.user import User
from apps.api.models.workspace import Workspace
from apps.api.schemas.graph import GraphBuildRequest, GraphBuildResponse, GraphRead

router = APIRouter(prefix="/workspaces/{workspace_id}/graphs", tags=["graphs"])


def _graph_read(graph: Graph) -> GraphRead:
    return GraphRead(
        id=graph.id,
        workspace_id=graph.workspace_id,
        status=graph.status,
        metadata=graph.metadata_json,
        created_at=graph.created_at,
    )


def _queue_graph_job(job_id: UUID, settings: Settings) -> tuple[bool, str | None]:
    try:
        from apps.worker.tasks.graph_build_task import build_graph

        if settings.celery_task_always_eager:
            build_graph(str(job_id))
            return True, None

        async_result = build_graph.delay(str(job_id))
        return True, str(async_result.id)
    except Exception as exc:
        if settings.celery_task_always_eager:
            raise
        return False, str(exc)


@router.post("", response_model=GraphBuildResponse, status_code=status.HTTP_201_CREATED)
async def create_graph(
    workspace_id: UUID,
    payload: GraphBuildRequest,
    session: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
    settings: Settings = Depends(get_config),
) -> GraphBuildResponse:
    workspace = await session.scalar(
        select(Workspace).where(Workspace.id == workspace_id, Workspace.owner_id == current_user.id)
    )
    if workspace is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Workspace not found")

    rows = await session.scalars(
        select(Document).where(Document.workspace_id == workspace.id, Document.id.in_(payload.document_ids))
    )
    documents = list(rows)
    if len(documents) != len(payload.document_ids):
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Invalid document IDs")

    graph = Graph(
        workspace_id=workspace.id,
        status="pending",
        metadata_json={
            "strategy": payload.strategy,
            "document_ids": [str(doc_id) for doc_id in payload.document_ids],
            "plan_strategy": payload.plan_strategy,
            "max_plan_actions": payload.max_plan_actions,
        },
    )
    session.add(graph)
    await session.flush()

    job = AsyncJob(
        workspace_id=workspace.id,
        job_type="graph_build",
        status="pending",
        payload={
            "graph_id": str(graph.id),
            "workspace_id": str(workspace.id),
            "document_ids": [str(doc_id) for doc_id in payload.document_ids],
            "strategy": payload.strategy,
            "keyword": payload.keyword,
            "metadata_field": payload.metadata_field,
            "metadata_value": payload.metadata_value,
            "plan_strategy": payload.plan_strategy,
            "max_plan_actions": payload.max_plan_actions,
        },
    )
    session.add(job)
    await session.flush()

    await session.commit()
    await session.refresh(graph)
    await session.refresh(job)

    queued, queue_error = _queue_graph_job(job.id, settings)
    if queue_error is not None:
        job.error = queue_error
        await session.commit()
        await session.refresh(job)

    return GraphBuildResponse(
        graph=_graph_read(graph),
        job_id=job.id,
        status=job.status,
        queued=queued,
    )


@router.get("/{graph_id}", response_model=GraphRead)
async def get_graph(
    workspace_id: UUID,
    graph_id: UUID,
    session: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
) -> GraphRead:
    workspace = await session.scalar(
        select(Workspace).where(Workspace.id == workspace_id, Workspace.owner_id == current_user.id)
    )
    if workspace is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Workspace not found")

    graph = await session.scalar(select(Graph).where(Graph.id == graph_id, Graph.workspace_id == workspace.id))
    if graph is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Graph not found")

    return _graph_read(graph)


@router.get("", response_model=list[GraphRead])
async def list_graphs(
    workspace_id: UUID,
    session: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
) -> list[GraphRead]:
    workspace = await session.scalar(
        select(Workspace).where(Workspace.id == workspace_id, Workspace.owner_id == current_user.id)
    )
    if workspace is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Workspace not found")

    rows = await session.scalars(
        select(Graph).where(Graph.workspace_id == workspace.id).order_by(Graph.created_at.desc())
    )
    return [_graph_read(graph) for graph in rows]
