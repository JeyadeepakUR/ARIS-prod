"""Plan generation task using persisted graph artifacts."""

from __future__ import annotations

import asyncio
from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime
from uuid import UUID

from sqlalchemy import delete, select

from apps.api.config import get_settings
from apps.api.core import database
from apps.api.models.edge import Edge
from apps.api.models.graph import Graph
from apps.api.models.job import AsyncJob
from apps.api.models.node import Node
from apps.api.models.plan import PlanAction
from apps.api.services.graph_service import GraphService
from apps.worker.main import celery_app


@celery_app.task(name="apps.worker.tasks.plan_task.generate_plan")
def generate_plan(job_id: str) -> None:
    """Generate plan actions from an existing persisted graph."""

    try:
        asyncio.get_running_loop()
    except RuntimeError:
        asyncio.run(_generate_plan_async(job_id))
        return

    with ThreadPoolExecutor(max_workers=1) as executor:
        future = executor.submit(asyncio.run, _generate_plan_async(job_id))
        future.result()


async def _generate_plan_async(job_id: str) -> None:
    settings = get_settings()
    database.init_database(settings)

    if database.SessionLocal is None:
        msg = "Database session factory is not initialized"
        raise RuntimeError(msg)

    graph_service = GraphService()
    async with database.SessionLocal() as session:
        job = await session.scalar(select(AsyncJob).where(AsyncJob.id == UUID(job_id)))
        if job is None:
            await database.dispose_database()
            return

        payload = job.payload
        graph_id = UUID(str(payload.get("graph_id")))
        graph = await session.scalar(select(Graph).where(Graph.id == graph_id))
        if graph is None:
            job.status = "failed"
            job.error = "Graph not found"
            job.completed_at = datetime.now(UTC)
            await session.commit()
            await database.dispose_database()
            return

        job.status = "processing"
        job.started_at = datetime.now(UTC)
        await session.commit()

        try:
            nodes = list(await session.scalars(select(Node).where(Node.graph_id == graph.id)))
            edges = list(await session.scalars(select(Edge).where(Edge.graph_id == graph.id)))
            actions = graph_service.create_plans_from_persisted(
                graph_id=graph.id,
                nodes=nodes,
                edges=edges,
                strategy=str(payload.get("strategy", "weak-evidence")),
                max_actions=int(payload.get("max_actions", 10)),
            )

            await session.execute(delete(PlanAction).where(PlanAction.graph_id == graph.id))
            for action in actions:
                session.add(
                    PlanAction(
                        id=action.action_id,
                        graph_id=graph.id,
                        action_type=action.action_type,
                        description=action.description,
                        evidence=action.evidence,
                        rationale=action.rationale,
                        priority=action.priority,
                        status="pending",
                        metadata_json={k: v for k, v in action.metadata.items()},
                    )
                )

            job.status = "ready"
            job.error = None
            job.result = {"graph_id": str(graph.id), "actions_count": len(actions)}
            job.trace_json = {
                "strategy": str(payload.get("strategy", "weak-evidence")),
                "max_actions": int(payload.get("max_actions", 10)),
                "actions_count": len(actions),
            }
        except Exception as exc:
            job.status = "failed"
            job.error = str(exc)
            job.result = {"graph_id": str(graph.id), "status": "failed"}
            job.trace_json = {"error": str(exc)}
        finally:
            job.completed_at = datetime.now(UTC)
            await session.commit()

    await database.dispose_database()
