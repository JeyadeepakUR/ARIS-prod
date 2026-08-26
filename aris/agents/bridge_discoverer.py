"""
Bridge Discoverer agent node.

Queries the DB for cross-domain node pairs with high cosine similarity
(pgvector self-join), then asks the LLM to validate and describe each bridge
mechanism. Also detects multi-hop bridges (A–C, C–B → A–B via C).

Changes from v1:
- Evidence passed to validation LLM is actual chunk text, not just node labels.
- Sibling-domain pairs (ML ↔ Deep Learning, ML ↔ RL, etc.) are hard-filtered
  before any LLM call — these are parent/child fields, not cross-domain bridges.
- Validation prompt is research-grade: requires a specific named mechanism and
  rejects generic category descriptions.
- BridgeEdge evidence_a/b stores real chunk text snippets for downstream
  hypothesis generation and the EdgeInspector "evidence side by side" panel.

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
from sqlalchemy import text as sa_text

from apps.api.models.edge import Edge
from aris.agents.state import BridgeEdge, ResearchState, StreamEvent
from aris.retrieval.vector_store import VectorStore

logger = logging.getLogger(__name__)

# ── Domain proximity guard ──────────────────────────────────────────────────
# These domain pairs are intellectually the same field or direct parent/child
# relationships. A bridge between them is trivially obvious and not useful.
_SIBLING_DOMAIN_PAIRS: frozenset[frozenset] = frozenset({
    frozenset({"Machine Learning", "Deep Learning"}),
    frozenset({"Machine Learning", "Reinforcement Learning"}),
    frozenset({"Deep Learning", "Reinforcement Learning"}),
    frozenset({"Machine Learning", "Statistics"}),
    frozenset({"Deep Learning", "Computer Vision"}),       # modern CV *is* deep learning
    frozenset({"Deep Learning", "Natural Language Processing"}),  # modern NLP *is* DL
})

# ── Prompts ──────────────────────────────────────────────────────────────────

_BRIDGE_VALIDATION_PROMPT = """\
You are a research intelligence system evaluating cross-domain knowledge bridges.

A bridge is VALID only if ALL of the following are true:
1. The two domains are genuinely different disciplines — not parent/child fields
   (e.g. "Machine Learning" and "Deep Learning" are the SAME field, not different).
2. The bridge_concept names a SPECIFIC, non-obvious transfer mechanism — not a
   generic category like "machine learning techniques" or "neural network methods".
3. A practitioner in Domain B would find the connection SURPRISING and useful.
4. The connection is supported by the actual evidence text below.

Concept A: "{label_a}"
Domain A: {domain_a}
Evidence from paper (Domain A):
  "{evidence_a}"

Concept B: "{label_b}"
Domain B: {domain_b}
Evidence from paper (Domain B):
  "{evidence_b}"

Embedding cosine similarity: {score:.2f}

REJECT if:
- Domains are same-family (ML/DL/RL/Statistics are all one family)
- Bridge concept would just repeat the concept names or use generic words
- Connection is obvious to anyone in either field
- Evidence does not actually support a specific transfer

VALID bridge_concept examples (specific, named, non-obvious):
  "Rényi divergence noise calibration for federated gradient privacy"
  "spectral graph clustering applied to protein interaction network analysis"
  "zero-knowledge proof compression for verifiable smart contract execution"
  "attention-based sequence alignment for genomic motif discovery"

INVALID bridge_concept examples (generic, reject these):
  "machine learning techniques"   — too generic
  "neural network methods"        — trivial, not a mechanism
  "deep learning approaches"      — category label, not a transfer mechanism
  "data processing methods"       — meaningless as a bridge

Return ONLY valid JSON:
{{
  "valid": true/false,
  "bridge_concept": "precise named transfer mechanism, 8-15 words (null if not valid)",
  "confidence": 0.0-1.0,
  "explanation": "one sentence: what SPECIFICALLY transfers, and why it is non-obvious to domain B practitioners"
}}"""

_MULTIHOP_PROMPT = """\
Three scientific concepts form an indirect cross-domain bridge:

Source:       "{label_a}" (Domain: {domain_a})
Intermediate: "{label_c}" (Domain: {domain_c})
Target:       "{label_b}" (Domain: {domain_b})

The A–C bridge mechanism: {bridge_ac}
The C–B bridge mechanism: {bridge_cb}

Describe the end-to-end multi-hop knowledge transfer path from {domain_a} to {domain_b}.
Focus on what a researcher in {domain_b} would actually gain.

Return ONLY:
{{
  "bridge_concept": "end-to-end transfer mechanism (8-15 words)",
  "confidence": 0.0-1.0,
  "explanation": "one sentence describing the full indirect link and its research value"
}}"""

# ── Tunables ─────────────────────────────────────────────────────────────────
_BRIDGE_MIN_SCORE = 0.50
_BRIDGE_LIMIT = 40
_MULTIHOP_MIN_INTERMEDIATE_SCORE = 0.50


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _stream_event(event_type: str, content: dict) -> StreamEvent:
    return {
        "agent_node": "bridge_discoverer",
        "event_type": event_type,
        "content": content,
        "timestamp": _now_iso(),
    }


async def _fetch_node_evidence(session, node_id: str) -> str:
    """
    Return up to two source-chunk text snippets for a concept node.

    Runs under a savepoint so any SQL failure rolls back only this sub-operation,
    leaving the outer transaction intact. Falls back to "" on any error.
    """
    from apps.api.models.node import Node as NodeModel
    from sqlalchemy import select as sa_select
    try:
        async with session.begin_nested():  # SAVEPOINT — protects outer transaction
            node = await session.scalar(
                sa_select(NodeModel).where(NodeModel.id == UUID(node_id))
            )
            if not node:
                return ""
            meta = node.metadata_json or {}
            chunk_ids = [str(c) for c in meta.get("source_chunk_ids", [])[:2]]
            if not chunk_ids:
                return ""
            placeholders = ", ".join(f":c{i}" for i in range(len(chunk_ids)))
            cr = await session.execute(
                sa_text(
                    "SELECT content FROM document_chunks "
                    f"WHERE id::text IN ({placeholders}) LIMIT 2"
                ),
                {f"c{i}": cid for i, cid in enumerate(chunk_ids)},
            )
            texts = [r.content for r in cr.fetchall() if r.content]
            return " [...] ".join(t[:450] for t in texts)
    except Exception as exc:
        logger.debug("_fetch_node_evidence(%s): %s", node_id, exc)
        return ""


async def bridge_discoverer_node(state: ResearchState, config: RunnableConfig) -> dict:
    cfg = config.get("configurable", {})
    session = cfg.get("session")
    llm_provider = cfg.get("llm_provider")

    graph_id: str = state["graph_id"]

    if not session or not llm_provider:
        return _error_return("Missing session or llm_provider in config")

    loop = asyncio.get_event_loop()
    vs = VectorStore(session)

    # ── 1. Candidate cross-domain pairs via pgvector self-join ────────────────
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

    # ── 2. Filter sibling-domain pairs before any LLM call ────────────────────
    non_trivial = []
    for row in candidates:
        pair = frozenset({row["domain1"], row["domain2"]})
        if pair in _SIBLING_DOMAIN_PAIRS:
            logger.debug(
                "bridge_discoverer: skipping sibling-domain pair %s ↔ %s",
                row["domain1"], row["domain2"],
            )
            continue
        non_trivial.append(row)

    logger.info(
        "bridge_discoverer: %d candidates remain after sibling-domain filter (%d removed)",
        len(non_trivial), len(candidates) - len(non_trivial),
    )

    # ── 3. Pre-fetch chunk evidence for all candidate nodes (cached) ──────────
    all_node_ids: set[str] = set()
    for row in non_trivial:
        all_node_ids.add(row["node1_id"])
        all_node_ids.add(row["node2_id"])

    evidence_cache: dict[str, str] = {}
    for nid in all_node_ids:
        evidence_cache[nid] = await _fetch_node_evidence(session, nid)

    # ── 4. LLM validation with real evidence ──────────────────────────────────
    validated_bridges: list[BridgeEdge] = []
    already_paired: set[frozenset[str]] = set()

    for row in non_trivial:
        pair_key = frozenset({row["node1_id"], row["node2_id"]})
        if pair_key in already_paired:
            continue

        evidence_a = evidence_cache.get(row["node1_id"]) or row["node1_label"]
        evidence_b = evidence_cache.get(row["node2_id"]) or row["node2_label"]

        prompt = _BRIDGE_VALIDATION_PROMPT.format(
            label_a=row["node1_label"],
            domain_a=row["domain1"],
            evidence_a=evidence_a[:400],
            label_b=row["node2_label"],
            domain_b=row["domain2"],
            evidence_b=evidence_b[:400],
            score=float(row["similarity"]),
        )
        try:
            result = await loop.run_in_executor(
                None,
                lambda p=prompt: llm_provider.complete_json(p, max_tokens=300),
            )
        except Exception as exc:
            logger.warning("bridge_discoverer: LLM validation failed — %s", exc)
            continue

        if not result.get("valid"):
            logger.debug(
                "bridge_discoverer: rejected '%s' ↔ '%s' — %s",
                row["node1_label"], row["node2_label"],
                result.get("explanation", "no reason given"),
            )
            continue

        bridge_concept = str(result.get("bridge_concept") or "").strip()[:255]
        if not bridge_concept or bridge_concept.lower() in {
            "null", "none", "n/a", "na", ""
        }:
            continue

        confidence = min(max(float(result.get("confidence", 0.7)), 0.0), 1.0)

        bridge: BridgeEdge = {
            "source_node_id": row["node1_id"],
            "target_node_id": row["node2_id"],
            "source_domain": row["domain1"],
            "target_domain": row["domain2"],
            "bridge_concept": bridge_concept,
            # Store actual chunk text — downstream agents (hypothesis_formulator)
            # and the EdgeInspector "evidence side by side" panel use these.
            "evidence_a": evidence_a[:500] if evidence_a else row["node1_label"],
            "evidence_b": evidence_b[:500] if evidence_b else row["node2_label"],
            "cosine_similarity": float(row["similarity"]),
            "confidence": confidence,
            "is_multi_hop": False,
            "intermediate_node_ids": [],
        }
        validated_bridges.append(bridge)
        already_paired.add(pair_key)

        await _persist_bridge_edge(session, graph_id, bridge, is_multi_hop=False)

        logger.info(
            "bridge_discoverer: validated bridge '%s' ↔ '%s' — \"%s\" (conf=%.2f)",
            row["node1_label"], row["node2_label"], bridge_concept, confidence,
        )

    # ── 5. Multi-hop inference: A–C + C–B → A–B via C ────────────────────────
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
                        bridge_ac=bridge_ac["bridge_concept"],
                        bridge_cb=bridge_cb["bridge_concept"],
                    )
                    result = await loop.run_in_executor(
                        None,
                        lambda p=prompt: llm_provider.complete_json(p, max_tokens=200),
                    )
                    bridge_concept = str(result.get("bridge_concept") or "").strip()[:255]
                    if not bridge_concept:
                        continue
                    confidence = min(float(result.get("confidence", 0.55)), 1.0) * 0.85

                    # Propagate chunk evidence from the direct bridges
                    ev_a = _get_evidence_from_bridge(bridge_ac, a_id)
                    ev_b = _get_evidence_from_bridge(bridge_cb, b_id)

                    mhop: BridgeEdge = {
                        "source_node_id": a_id,
                        "target_node_id": b_id,
                        "source_domain": domain_a,
                        "target_domain": domain_b,
                        "bridge_concept": bridge_concept,
                        "evidence_a": ev_a[:500],
                        "evidence_b": ev_b[:500],
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

    # ── 6. Intra-domain similarity edges (graph structure scaffolding) ─────────
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
            "candidates_evaluated": len(non_trivial),
            "sibling_pairs_filtered": len(candidates) - len(non_trivial),
            "total_edges": len(all_bridges),
        },
    )
    return {
        "bridges": all_bridges,
        "completed_steps": ["bridge_discoverer"],
        "stream_events": [event],
    }


async def _persist_bridge_edge(
    session, graph_id: str, bridge: BridgeEdge, *, is_multi_hop: bool
) -> None:
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
        return bridge["evidence_a"].split(" [...] ")[0][:80] or bridge["source_domain"]
    return bridge["evidence_b"].split(" [...] ")[0][:80] or bridge["target_domain"]


def _get_evidence_from_bridge(bridge: BridgeEdge, node_id: str) -> str:
    if bridge["source_node_id"] == node_id:
        return bridge["evidence_a"]
    return bridge["evidence_b"]


def _get_intermediate_label(bridge_ac: BridgeEdge, bridge_cb: BridgeEdge, node_c: str) -> str:
    if bridge_ac["source_node_id"] == node_c:
        return bridge_ac["evidence_a"].split(" [...] ")[0][:80]
    return bridge_ac["evidence_b"].split(" [...] ")[0][:80]


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
