"""
Bridge Discoverer agent node.

Queries the DB for cross-domain node pairs with high cosine similarity
(pgvector self-join), then asks the LLM to validate and describe each bridge
mechanism. Also detects multi-hop bridges (A–C, C–B → A–B via C).

Dependencies injected through LangGraph config["configurable"]:
    session      — AsyncSession
    llm_provider — LLMProvider
"""
from __future__ import annotations

import asyncio
import logging
from datetime import datetime, timezone
from uuid import UUID, uuid4

from langchain_core.runnables import RunnableConfig
from sqlalchemy import text

from apps.api.models.edge import Edge
from aris.agents.state import BridgeEdge, ResearchState, StreamEvent
from aris.retrieval.vector_store import VectorStore

logger = logging.getLogger(__name__)

_BRIDGE_VALIDATION_PROMPT = """\
Two scientific concepts from different research domains have been found to be \
semantically similar.

Concept A: "{label_a}" (Domain: {domain_a})
Concept B: "{label_b}" (Domain: {domain_b})
Cosine similarity: {score:.2f}

Is there a meaningful research bridge between these concepts — a principle, \
mechanism, or technique that enables knowledge transfer across the two domains?

Return ONLY a JSON object:
{{
  "valid": true/false,
  "bridge_concept": "precise transfer mechanism (5-10 words, null if not valid)",
  "confidence": 0.0-1.0,
  "explanation": "one sentence"
}}"""

_MULTIHOP_PROMPT = """\
Three concepts form an indirect bridge across research domains:

Source: "{label_a}" (Domain: {domain_a})
Intermediate: "{label_c}" (Domain: {domain_c})
Target: "{label_b}" (Domain: {domain_b})

Describe the multi-hop knowledge transfer path.

Return ONLY JSON:
{{
  "bridge_concept": "overall transfer mechanism (5-10 words)",
  "confidence": 0.0-1.0,
  "explanation": "one sentence describing the indirect link"
}}"""

# Free-tier rate-limit aware defaults — fewer LLM calls per build.
_BRIDGE_MIN_SCORE = 0.62
_BRIDGE_LIMIT = 20
_MULTIHOP_MIN_INTERMEDIATE_SCORE = 0.60


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _stream_event(event_type: str, content: dict) -> StreamEvent:
    return {
        "agent_node": "bridge_discoverer",
        "event_type": event_type,
        "content": content,
        "timestamp": _now_iso(),
    }


async def bridge_discoverer_node(state: ResearchState, config: RunnableConfig) -> dict:
    cfg = config.get("configurable", {})
    session = cfg.get("session")
    llm_provider = cfg.get("llm_provider")

    graph_id: str = state["graph_id"]

    if not session or not llm_provider:
        return _error_return("Missing session or llm_provider in config")

    loop = asyncio.get_event_loop()
    vs = VectorStore(session)

    # ── 1. Find candidate cross-domain pairs via pgvector self-join ───────────
    try:
        candidates = await vs.cross_domain_bridge_search(
            graph_id=graph_id,
            min_score=_BRIDGE_MIN_SCORE,
            limit=_BRIDGE_LIMIT,
        )
    except Exception as exc:
        logger.error("bridge_discoverer: vector search failed — %s", exc)
        return _error_return(str(exc))

    logger.info(
        "bridge_discoverer: %d cross-domain candidates (min_score=%.2f) for graph %s",
        len(candidates), _BRIDGE_MIN_SCORE, graph_id,
    )
    if candidates:
        domains_seen = {(r["domain1"], r["domain2"]) for r in candidates}
        logger.info("bridge_discoverer: domain pairs — %s", domains_seen)

    # ── 2. Validate each candidate with LLM ───────────────────────────────────
    validated_bridges: list[BridgeEdge] = []
    already_paired: set[frozenset[str]] = set()

    for row in candidates:
        pair_key = frozenset({row["node1_id"], row["node2_id"]})
        if pair_key in already_paired:
            continue

        prompt = _BRIDGE_VALIDATION_PROMPT.format(
            label_a=row["node1_label"],
            domain_a=row["domain1"],
            label_b=row["node2_label"],
            domain_b=row["domain2"],
            score=float(row["similarity"]),
        )
        try:
            result = await loop.run_in_executor(
                None,
                lambda p=prompt: llm_provider.complete_json(p, max_tokens=250),
            )
        except Exception as exc:
            logger.warning("bridge_discoverer: LLM validation failed — %s", exc)
            continue

        if not result.get("valid"):
            continue

        bridge_concept = str(result.get("bridge_concept") or "").strip()[:255] or "cross-domain link"
        confidence = min(max(float(result.get("confidence", 0.7)), 0.0), 1.0)

        bridge: BridgeEdge = {
            "source_node_id": row["node1_id"],
            "target_node_id": row["node2_id"],
            "source_domain": row["domain1"],
            "target_domain": row["domain2"],
            "bridge_concept": bridge_concept,
            "evidence_a": row["node1_label"],
            "evidence_b": row["node2_label"],
            "cosine_similarity": float(row["similarity"]),
            "confidence": confidence,
            "is_multi_hop": False,
            "intermediate_node_ids": [],
        }
        validated_bridges.append(bridge)
        already_paired.add(pair_key)

        # Persist edge
        await _persist_bridge_edge(session, graph_id, bridge, is_multi_hop=False)

    # ── 3. Multi-hop: A–C validated AND C–B validated → A–B multi-hop ─────────
    direct_pairs = {
        frozenset({b["source_node_id"], b["target_node_id"]}): b
        for b in validated_bridges
    }
    multihop_bridges: list[BridgeEdge] = []
    multihop_added: set[frozenset[str]] = set()

    nodes_in_bridges: set[str] = set()
    for b in validated_bridges:
        nodes_in_bridges.add(b["source_node_id"])
        nodes_in_bridges.add(b["target_node_id"])

    for node_c in nodes_in_bridges:
        # Find all bridges that include node_c
        bridges_with_c = [
            b for b in validated_bridges
            if b["source_node_id"] == node_c or b["target_node_id"] == node_c
        ]
        for i, bridge_ac in enumerate(bridges_with_c):
            for bridge_cb in bridges_with_c[i + 1:]:
                a_id = (
                    bridge_ac["target_node_id"]
                    if bridge_ac["source_node_id"] == node_c
                    else bridge_ac["source_node_id"]
                )
                b_id = (
                    bridge_cb["target_node_id"]
                    if bridge_cb["source_node_id"] == node_c
                    else bridge_cb["source_node_id"]
                )
                if a_id == b_id:
                    continue

                domain_a = (
                    bridge_ac["target_domain"]
                    if bridge_ac["source_node_id"] == node_c
                    else bridge_ac["source_domain"]
                )
                domain_b = (
                    bridge_cb["target_domain"]
                    if bridge_cb["source_node_id"] == node_c
                    else bridge_cb["source_domain"]
                )
                if domain_a == domain_b:
                    continue

                pair_key = frozenset({a_id, b_id})
                if pair_key in direct_pairs or pair_key in multihop_added:
                    continue

                # Ask LLM for the multihop mechanism
                try:
                    node_a_label = _get_label_from_bridge(bridge_ac, a_id)
                    node_b_label = _get_label_from_bridge(bridge_cb, b_id)
                    node_c_label = _get_intermediate_label(bridge_ac, bridge_cb, node_c)
                    domain_c = _get_intermediate_domain(bridge_ac, node_c)

                    prompt = _MULTIHOP_PROMPT.format(
                        label_a=node_a_label,
                        domain_a=domain_a,
                        label_c=node_c_label,
                        domain_c=domain_c,
                        label_b=node_b_label,
                        domain_b=domain_b,
                    )
                    result = await loop.run_in_executor(
                        None,
                        lambda p=prompt: llm_provider.complete_json(p, max_tokens=200),
                    )
                    bridge_concept = str(result.get("bridge_concept") or "").strip()[:255]
                    confidence = min(float(result.get("confidence", 0.55)), 1.0) * 0.85

                    mhop: BridgeEdge = {
                        "source_node_id": a_id,
                        "target_node_id": b_id,
                        "source_domain": domain_a,
                        "target_domain": domain_b,
                        "bridge_concept": bridge_concept or "indirect cross-domain link",
                        "evidence_a": node_a_label,
                        "evidence_b": node_b_label,
                        "cosine_similarity": (
                            bridge_ac["cosine_similarity"] + bridge_cb["cosine_similarity"]
                        ) / 2,
                        "confidence": confidence,
                        "is_multi_hop": True,
                        "intermediate_node_ids": [node_c],
                    }
                    multihop_bridges.append(mhop)
                    multihop_added.add(pair_key)
                    await _persist_bridge_edge(session, graph_id, mhop, is_multi_hop=True)
                except Exception as exc:
                    logger.warning("bridge_discoverer: multi-hop LLM failed — %s", exc)

    # ── 4. Intra-domain similarity edges (always run, gives the graph structure
    #        even when all concepts share one domain) ───────────────────────────
    intra_edges: list[BridgeEdge] = []
    try:
        intra_candidates = await vs.intra_domain_similarity_search(
            graph_id=graph_id,
            min_score=0.75,
            limit=80,
        )
        logger.info("bridge_discoverer: %d intra-domain candidates", len(intra_candidates))
        existing_pairs = {
            frozenset({b["source_node_id"], b["target_node_id"]})
            for b in validated_bridges + multihop_bridges
        }
        for row in intra_candidates:
            pair_key = frozenset({row["node1_id"], row["node2_id"]})
            if pair_key in existing_pairs:
                continue
            existing_pairs.add(pair_key)
            bridge: BridgeEdge = {
                "source_node_id": row["node1_id"],
                "target_node_id": row["node2_id"],
                "source_domain": row["domain1"],
                "target_domain": row["domain2"],
                "bridge_concept": f"semantic similarity within {row['domain1']}",
                "evidence_a": row["node1_label"],
                "evidence_b": row["node2_label"],
                "cosine_similarity": float(row["similarity"]),
                "confidence": float(row["similarity"]) * 0.9,
                "is_multi_hop": False,
                "intermediate_node_ids": [],
            }
            intra_edges.append(bridge)
            session.add(
                Edge(
                    id=uuid4(),
                    graph_id=UUID(graph_id),
                    source_node_id=UUID(bridge["source_node_id"]),
                    target_node_id=UUID(bridge["target_node_id"]),
                    edge_type="semantic_similarity",
                    edge_category="INTRA_DOMAIN",
                    bridge_concept=bridge["bridge_concept"],
                    evidence=(
                        f"[A] {bridge['evidence_a'][:500]}\n[B] {bridge['evidence_b'][:500]}"
                    ),
                    reasoning_trace_id=uuid4(),
                    confidence=bridge["confidence"],
                    metadata_json={
                        "domain": row["domain1"],
                        "cosine_similarity": bridge["cosine_similarity"],
                        "is_multi_hop": False,
                        "intermediate_node_ids": [],
                    },
                )
            )
    except Exception as exc:
        logger.warning("bridge_discoverer: intra-domain search failed — %s", exc)

    await session.flush()
    all_bridges = validated_bridges + multihop_bridges + intra_edges

    logger.info(
        "bridge_discoverer: %d cross-domain + %d multi-hop + %d intra-domain = %d total edges",
        len(validated_bridges), len(multihop_bridges), len(intra_edges), len(all_bridges),
    )

    event = _stream_event(
        "step_complete",
        {
            "bridges_found": len(validated_bridges),
            "multihop_bridges": len(multihop_bridges),
            "intra_domain_edges": len(intra_edges),
            "candidates_evaluated": len(candidates),
            "total_edges": len(all_bridges),
        },
    )
    return {
        "bridges": all_bridges,
        "completed_steps": ["bridge_discoverer"],
        "stream_events": [event],
    }


async def _persist_bridge_edge(session, graph_id: str, bridge: BridgeEdge, *, is_multi_hop: bool) -> None:
    session.add(
        Edge(
            id=uuid4(),
            graph_id=UUID(graph_id),
            source_node_id=UUID(bridge["source_node_id"]),
            target_node_id=UUID(bridge["target_node_id"]),
            edge_type="cross_domain_bridge",
            edge_category="INTER_DOMAIN_BRIDGE",
            bridge_concept=bridge["bridge_concept"],
            evidence=(
                f"[A] {bridge['evidence_a'][:500]}\n[B] {bridge['evidence_b'][:500]}"
            ),
            reasoning_trace_id=uuid4(),
            confidence=bridge["confidence"],
            metadata_json={
                "source_domain": bridge["source_domain"],
                "target_domain": bridge["target_domain"],
                "cosine_similarity": bridge["cosine_similarity"],
                "is_multi_hop": is_multi_hop,
                "intermediate_node_ids": bridge["intermediate_node_ids"],
            },
        )
    )


def _get_label_from_bridge(bridge: BridgeEdge, node_id: str) -> str:
    if bridge["source_node_id"] == node_id:
        return bridge["evidence_a"]
    return bridge["evidence_b"]


def _get_intermediate_label(bridge_ac: BridgeEdge, bridge_cb: BridgeEdge, node_c: str) -> str:
    if bridge_ac["source_node_id"] == node_c:
        return bridge_ac["evidence_a"]
    return bridge_ac["evidence_b"]


def _get_intermediate_domain(bridge: BridgeEdge, node_c: str) -> str:
    if bridge["source_node_id"] == node_c:
        return bridge["source_domain"]
    return bridge["target_domain"]


def _error_return(msg: str) -> dict:
    return {
        "bridges": [],
        "completed_steps": ["bridge_discoverer"],
        "error": msg,
        "stream_events": [{
            "agent_node": "bridge_discoverer",
            "event_type": "error",
            "content": {"error": msg},
            "timestamp": _now_iso(),
        }],
    }
