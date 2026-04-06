"""Plan routes for generating and listing research actions."""

from __future__ import annotations

from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from apps.api.config import Settings
from apps.api.dependencies import get_config, get_current_user, get_db
from apps.api.models.graph import Graph
from apps.api.models.job import AsyncJob
from apps.api.models.plan import PlanAction
from apps.api.models.user import User
from apps.api.models.workspace import Workspace
from apps.api.schemas.plan import PlanActionRead, PlanGenerateRequest, PlanGenerateResponse

router = APIRouter(prefix="/graphs/{graph_id}/plans", tags=["plans"])


def _queue_plan_job(job_id: UUID, settings: Settings) -> tuple[bool, str | None]:
    try:
        from apps.worker.tasks.plan_task import generate_plan

        if settings.celery_task_always_eager:
            generate_plan(str(job_id))
            return True, None

        async_result = generate_plan.delay(str(job_id))
        return True, str(async_result.id)
    except Exception as exc:
        if settings.celery_task_always_eager:
            raise
        return False, str(exc)


@router.post("/generate", response_model=PlanGenerateResponse, status_code=status.HTTP_202_ACCEPTED)
async def generate_graph_plan(
    graph_id: UUID,
    payload: PlanGenerateRequest,
    session: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
    settings: Settings = Depends(get_config),
) -> PlanGenerateResponse:
    graph = await session.scalar(select(Graph).where(Graph.id == graph_id))
    if graph is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Graph not found")

    workspace = await session.scalar(
        select(Workspace).where(Workspace.id == graph.workspace_id, Workspace.owner_id == current_user.id)
    )
    if workspace is None:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Forbidden")

    job = AsyncJob(
        workspace_id=workspace.id,
        job_type="plan_generate",
        status="pending",
        payload={
            "graph_id": str(graph.id),
            "strategy": payload.strategy,
            "max_actions": payload.max_actions,
        },
    )
    session.add(job)
    await session.flush()

    await session.commit()
    await session.refresh(job)

    queued, queue_error = _queue_plan_job(job.id, settings)
    if queue_error is not None:
        job.error = queue_error
        await session.commit()
        await session.refresh(job)

    return PlanGenerateResponse(graph_id=graph.id, job_id=job.id, status=job.status, queued=queued)


@router.get("", response_model=list[PlanActionRead])
async def list_graph_plans(
    graph_id: UUID,
    session: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
) -> list[PlanActionRead]:
    graph = await session.scalar(select(Graph).where(Graph.id == graph_id))
    if graph is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Graph not found")

    workspace = await session.scalar(
        select(Workspace).where(Workspace.id == graph.workspace_id, Workspace.owner_id == current_user.id)
    )
    if workspace is None:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Forbidden")

    rows = await session.scalars(
        select(PlanAction).where(PlanAction.graph_id == graph.id).order_by(PlanAction.priority.desc())
    )
    return [
        PlanActionRead(
            id=action.id,
            graph_id=action.graph_id,
            action_type=action.action_type,
            description=action.description,
            evidence=action.evidence,
            rationale=action.rationale,
            priority=action.priority,
            status=action.status,
            metadata=action.metadata_json,
            created_at=action.created_at,
        )
        for action in rows
    ]
