"""Generate hypotheses from inter-domain bridge edges using LLM synthesis."""

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
from aris.llm.provider import get_provider


_HYPOTHESIS_PROMPT = """\
You are a research intelligence system generating falsifiable scientific hypotheses.

A knowledge graph has identified a cross-domain bridge between two research areas.

Source node: {source_label}
Source domain: {source_domain}
Target node: {target_label}
Target domain: {target_domain}
Bridge concept: {bridge_concept}
Edge evidence: {evidence}

Generate a specific, testable, falsifiable research hypothesis about this cross-domain connection.

Return a JSON object with:
- "statement": the primary hypothesis (1-2 sentences, specific and falsifiable)
- "null_hypothesis": the null hypothesis to reject
- "hypothesis_type": one of "causal", "correlational", "technology_transfer", "mechanistic"
- "methodology_hint": brief suggested experimental approach (1 sentence)
- "evidence_basis": what in the evidence supports this hypothesis

Return only the JSON object."""


def _generate_hypothesis_with_llm(
    source: Node,
    target: Node,
    edge: Edge,
    provider,
) -> str:
    bridge = edge.bridge_concept or "cross-domain transfer mechanism"
    evidence_snippet = " ".join(edge.evidence.split())[:350]

    try:
        prompt = _HYPOTHESIS_PROMPT.format(
            source_label=source.label,
            source_domain=source.cluster_id or "unknown domain",
            target_label=target.label,
            target_domain=target.cluster_id or "unknown domain",
            bridge_concept=bridge,
            evidence=evidence_snippet,
        )
        result = provider.complete_json(prompt, max_tokens=400)
        statement = str(result.get("statement", "")).strip()
        null_hyp = str(result.get("null_hypothesis", "")).strip()
        hyp_type = str(result.get("hypothesis_type", "causal")).strip()
        methodology = str(result.get("methodology_hint", "")).strip()
        evidence_basis = str(result.get("evidence_basis", "")).strip()

        if not statement:
            return _generate_hypothesis_fallback(source, target, edge)

        parts = [f"Hypothesis ({hyp_type}): {statement}"]
        if null_hyp:
            parts.append(f"Null hypothesis: {null_hyp}")
        if methodology:
            parts.append(f"Methodology: {methodology}")
        if evidence_basis:
            parts.append(f"Evidence basis: {evidence_basis}")
        return "\n".join(parts)
    except Exception:
        return _generate_hypothesis_fallback(source, target, edge)


def _generate_hypothesis_fallback(source: Node, target: Node, edge: Edge) -> str:
    bridge = edge.bridge_concept or "cross-domain transfer mechanism"
    evidence_snippet = " ".join(edge.evidence.split())[:200]
    return (
        f"Hypothesis (technology_transfer): Applying {bridge} developed in "
        f"{source.cluster_id or 'the source domain'} ({source.label}) to "
        f"{target.cluster_id or 'the target domain'} ({target.label}) "
        f"will yield measurable performance improvement on benchmark tasks.\n"
        f"Null hypothesis: {bridge} from {source.cluster_id or 'source'} "
        f"does not improve outcomes in {target.cluster_id or 'target'}.\n"
        f"Evidence basis: {evidence_snippet}"
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

    provider = get_provider(
        settings.llm_provider,
        model=settings.llm_model,
        api_key=settings.llm_api_key,
        base_url=settings.llm_base_url,
    )

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

            hypothesis_text = _generate_hypothesis_with_llm(source, target, edge, provider)
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
