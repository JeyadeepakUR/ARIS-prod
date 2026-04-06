"""Graph build task for Sprint 3 ARIS pipeline integration."""

from __future__ import annotations

import asyncio
from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime
from uuid import UUID

from sqlalchemy import select

from apps.api.config import get_settings
from apps.api.core import database
from apps.api.models.document import Document
from apps.api.models.edge import Edge
from apps.api.models.graph import Graph
from apps.api.models.job import AsyncJob
from apps.api.models.node import Node
from apps.api.models.plan import PlanAction
from apps.api.services.graph_service import GraphService
from apps.api.tasks.hypothesis_task import generate_hypotheses
from apps.worker.main import celery_app


@celery_app.task(name="apps.worker.tasks.graph_build_task.build_graph")
def build_graph(job_id: str) -> None:
    """Build and persist graph nodes/edges/plans from ready documents."""

    try:
        asyncio.get_running_loop()
    except RuntimeError:
        asyncio.run(_build_graph_async(job_id))
        return

    with ThreadPoolExecutor(max_workers=1) as executor:
        future = executor.submit(asyncio.run, _build_graph_async(job_id))
        future.result()


async def _build_graph_async(job_id: str) -> None:
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
        workspace_id = UUID(str(payload.get("workspace_id")))
        document_ids = [UUID(str(value)) for value in payload.get("document_ids", [])]

        graph = await session.scalar(select(Graph).where(Graph.id == graph_id, Graph.workspace_id == workspace_id))
        if graph is None:
            job.status = "failed"
            job.error = "Graph not found"
            job.completed_at = datetime.now(UTC)
            await session.commit()
            await database.dispose_database()
            return

        job.status = "processing"
        job.started_at = datetime.now(UTC)
        graph.status = "processing"
        await session.commit()

        try:
            rows = await session.scalars(
                select(Document).where(Document.workspace_id == workspace_id, Document.id.in_(document_ids))
            )
            documents_by_id = {row.id: row for row in rows}
            ordered_documents = [documents_by_id[doc_id] for doc_id in document_ids if doc_id in documents_by_id]

            if len(ordered_documents) != len(document_ids):
                raise ValueError("One or more documents do not belong to workspace")

            if any(document.status != "ready" for document in ordered_documents):
                raise ValueError("All documents must be ready before graph build")

            artifacts = graph_service.build_graph(
                ordered_documents,
                strategy=str(payload.get("strategy", "sequential")),
                keyword=_optional_string(payload.get("keyword")),
                metadata_field=_optional_string(payload.get("metadata_field")),
                metadata_value=_optional_string(payload.get("metadata_value")),
                plan_strategy=str(payload.get("plan_strategy", "weak-evidence")),
                max_plan_actions=int(payload.get("max_plan_actions", 10)),
            )

            document_to_node = {}
            for node in artifacts.graph.nodes:
                document_to_node[node.document_id] = node.node_id
                session.add(
                    Node(
                        id=node.node_id,
                        graph_id=graph.id,
                        document_id=node.document_id,
                        label=node.label,
                        node_type=node.node_type,
                        metadata_json={k: v for k, v in node.metadata.items()},
                    )
                )

            for edge in artifacts.graph.edges:
                source_node_id = document_to_node.get(edge.source_id, edge.source_id)
                target_node_id = document_to_node.get(edge.target_id, edge.target_id)
                session.add(
                    Edge(
                        id=edge.edge_id,
                        graph_id=graph.id,
                        source_node_id=source_node_id,
                        target_node_id=target_node_id,
                        edge_type=edge.edge_type,
                        evidence=edge.evidence,
                        reasoning_trace_id=edge.reasoning_trace_id,
                        confidence=edge.confidence,
                        metadata_json={k: v for k, v in edge.metadata.items()},
                    )
                )

            await session.flush()
            persisted_nodes = list(await session.scalars(select(Node).where(Node.graph_id == graph.id)))
            persisted_edges = list(await session.scalars(select(Edge).where(Edge.graph_id == graph.id)))
            await graph_service.classify_nodes_and_edges(graph.id, persisted_nodes, persisted_edges, session)

            for action in artifacts.plan_actions:
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

            graph.status = "ready"
            graph.metadata_json = {
                **graph.metadata_json,
                "nodes_count": len(artifacts.graph.nodes),
                "edges_count": len(artifacts.graph.edges),
            }

            job.status = "ready"
            job.error = None
            job.result = {
                "graph_id": str(graph.id),
                "nodes_count": len(artifacts.graph.nodes),
                "edges_count": len(artifacts.graph.edges),
                "plan_actions_count": len(artifacts.plan_actions),
            }
            job.trace_json = artifacts.trace
        except Exception as exc:
            graph.status = "failed"
            job.status = "failed"
            job.error = str(exc)
            job.result = {"graph_id": str(graph.id), "status": "failed"}
            job.trace_json = {"error": str(exc)}
        finally:
            job.completed_at = datetime.now(UTC)
            await session.commit()

            if job.status == "ready":
                generate_hypotheses.si(str(graph.id)).delay()

    await database.dispose_database()


def _optional_string(value: object) -> str | None:
    if isinstance(value, str) and value.strip():
        return value.strip()
    return None
