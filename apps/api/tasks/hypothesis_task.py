"""Generate hypotheses from inter-domain bridge edges."""

from __future__ import annotations

import asyncio
from concurrent.futures import ThreadPoolExecutor
from uuid import UUID, uuid4

from sqlalchemy import select

from apps.api.config import get_settings
from apps.api.core import database
from apps.api.models.edge import Edge
from apps.api.models.hypothesis import Hypothesis
from apps.api.models.node import Node
from apps.worker.main import celery_app


def _summarize_evidence(text: str, max_chars: int) -> str:
    return " ".join(text.split())[:max_chars]


def _generate_hypothesis_text(source: Node, target: Node, edge: Edge) -> str:
    bridge = edge.bridge_concept or "cross-domain transfer mechanism"
    evidence = _summarize_evidence(edge.evidence, 180)
    return (
        f"If {source.label} is optimized within {source.cluster_id or 'its source domain'}, "
        f"then {target.label} in {target.cluster_id or 'the target domain'} shows measurable improvement, "
        f"because {bridge} mediates the observed transfer pattern from evidence: {evidence}."
    )


@celery_app.task(name="tasks.generate_hypotheses", bind=True, max_retries=2)
def generate_hypotheses(self, graph_id: str) -> dict:
    """Generate one hypothesis per high-confidence inter-domain bridge edge."""

    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(_generate_hypotheses_async(graph_id))

    with ThreadPoolExecutor(max_workers=1) as executor:
        future = executor.submit(asyncio.run, _generate_hypotheses_async(graph_id))
        return future.result()


async def _generate_hypotheses_async(graph_id: str) -> dict:
    settings = get_settings()
    database.init_database(settings)

    if database.SessionLocal is None:
        msg = "Database session factory is not initialized"
        raise RuntimeError(msg)

    created_count = 0
    skipped_count = 0
    graph_uuid = UUID(graph_id)

    async with database.SessionLocal() as session:
        bridge_edges = list(
            await session.scalars(
                select(Edge).where(
                    Edge.graph_id == graph_uuid,
                    Edge.edge_category == "INTER_DOMAIN_BRIDGE",
                    Edge.confidence >= 0.7,
                )
            )
        )

        if not bridge_edges:
            await database.dispose_database()
            return {"graph_id": graph_id, "created": 0, "skipped": 0}

        node_ids = {edge.source_node_id for edge in bridge_edges} | {edge.target_node_id for edge in bridge_edges}
        nodes = list(await session.scalars(select(Node).where(Node.id.in_(node_ids))))
        nodes_by_id = {node.id: node for node in nodes}

        for edge in bridge_edges:
            existing = await session.scalar(select(Hypothesis).where(Hypothesis.edge_id == edge.id))
            if existing is not None:
                skipped_count += 1
                continue

            source = nodes_by_id.get(edge.source_node_id)
            target = nodes_by_id.get(edge.target_node_id)
            if source is None or target is None:
                skipped_count += 1
                continue

            hypothesis_text = _generate_hypothesis_text(source, target, edge)
            session.add(
                Hypothesis(
                    id=uuid4(),
                    graph_id=graph_uuid,
                    edge_id=edge.id,
                    hypothesis_text=hypothesis_text,
                    confidence=edge.confidence,
                    status="proposed",
                )
            )
            created_count += 1

        await session.commit()

    await database.dispose_database()
    return {"graph_id": graph_id, "created": created_count, "skipped": skipped_count}
