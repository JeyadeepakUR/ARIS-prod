"""
Hypothesis Formulator agent node.

For each confirmed cross-domain bridge, generates a testable research hypothesis
grounded in actual paper evidence (chunk text fetched from the DB), not just
concept labels.

Changes from v1:
- Fetches real source-chunk text for both bridge endpoints before LLM call.
- Prompt requires specific, quantitative, falsifiable hypotheses — not generic
  "applying X to Y will improve performance" templates.
- Prompt explicitly bans template language and requires grounding in the evidence.

Dependencies injected through LangGraph config["configurable"]:
    session      — AsyncSession
    embedder     — Embedder (for novelty check)
    llm_provider — LLMProvider
"""
from __future__ import annotations

import asyncio
import logging
from datetime import datetime, timezone
from uuid import UUID

from langchain_core.runnables import RunnableConfig
from sqlalchemy import text as sa_text

from aris.agents.state import BridgeEdge, Hypothesis, ResearchState, StreamEvent
from aris.retrieval.vector_store import VectorStore

logger = logging.getLogger(__name__)

_HYPOTHESIS_PROMPT = """\
You are a research scientist generating a novel, publishable research hypothesis \
from a validated cross-domain knowledge bridge.

BRIDGE:
Source concept: "{source_label}"  (Domain: {source_domain})
Target concept:  "{target_label}" (Domain: {target_domain})
Transfer mechanism: {bridge_concept}

RAW EVIDENCE FROM THE PAPERS:

[Paper A — {source_domain}]
{evidence_a}

[Paper B — {target_domain}]
{evidence_b}

YOUR TASK:
Generate ONE specific, testable research hypothesis that exploits this bridge.

STRICT REQUIREMENTS — your hypothesis MUST satisfy all of these:
1. SPECIFIC: Name the exact algorithm, metric, dataset, or experimental condition.
   Do NOT write "method X" — write "LSTM with attention over 30-day sliding windows".
2. FALSIFIABLE: State a concrete pass/fail criterion or a measurable prediction.
3. GROUNDED: Reference specific claims, methods, or numbers from the evidence above.
4. QUANTITATIVE where possible: "within 2% of", "improves F1 by ≥ 5 points", etc.
5. NON-OBVIOUS: The hypothesis must be something a practitioner in {target_domain}
   would not already assume.

BANNED template phrases (reject your own response if it contains these):
- "will yield measurable performance improvement"
- "will improve outcomes"
- "applying X to Y will improve Z"
- "can enhance performance"
- Any sentence that could fit ANY pair of domains without modification.

GOOD hypothesis example:
  "We hypothesize that replacing standard softmax cross-entropy in IDS classifiers \
with focal loss (γ=2) — originally developed for class-imbalanced object detection — \
will increase recall on rare intrusion categories in the UNSW-NB15 benchmark by ≥8 \
percentage points while keeping precision degradation below 3%, because the focal \
loss term down-weights easy benign-traffic examples that dominate training."

BAD hypothesis example (do not produce this):
  "We hypothesize that applying attention mechanisms from NLP to cybersecurity \
will yield measurable performance improvement on benchmark tasks."

Return ONLY valid JSON:
{{
  "statement": "We hypothesize that... (2-4 sentences, specific, grounded, quantitative)",
  "null_hypothesis": "There is no statistically significant difference in [specific metric] \
when [specific intervention] is applied to [specific setting] (α=0.05)",
  "hypothesis_type": "causal" | "correlational" | "technology_transfer" | "mechanistic",
  "methodology_hint": "Baseline: [X]. Dataset: [Y]. Intervention: [Z]. \
Primary metric: [M]. Comparison: [describe controlled comparison] (2-3 sentences)",
  "key_prediction": "The single most falsifiable quantitative or structural prediction",
  "testability_score": 0.0-1.0,
  "confidence": 0.0-1.0
}}"""

_NOVELTY_THRESHOLD = 0.85


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _stream_event(event_type: str, content: dict) -> StreamEvent:
    return {
        "agent_node": "hypothesis_formulator",
        "event_type": event_type,
        "content": content,
        "timestamp": _now_iso(),
    }


async def _fetch_chunk_evidence(session, node_id: str, max_chars: int = 600) -> str:
    """
    Return actual chunk text for a concept node's source chunks.

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
            chunk_ids = [str(c) for c in meta.get("source_chunk_ids", [])[:3]]
            if not chunk_ids:
                return node.label or ""
            placeholders = ", ".join(f":c{i}" for i in range(len(chunk_ids)))
            cr = await session.execute(
                sa_text(
                    "SELECT content FROM document_chunks "
                    f"WHERE id::text IN ({placeholders}) LIMIT 3"
                ),
                {f"c{i}": cid for i, cid in enumerate(chunk_ids)},
            )
            texts = [r.content for r in cr.fetchall() if r.content]
            if not texts:
                return node.label or ""
            combined = "\n[...]\n".join(t[: max_chars // len(texts)] for t in texts)
            return combined[:max_chars]
    except Exception as exc:
        logger.debug("_fetch_chunk_evidence(%s): %s", node_id, exc)
        return ""


async def hypothesis_formulator_node(state: ResearchState, config: RunnableConfig) -> dict:
    cfg = config.get("configurable", {})
    session = cfg.get("session")
    embedder = cfg.get("embedder")
    llm_provider = cfg.get("llm_provider")

    bridges: list[BridgeEdge] = state.get("bridges", [])

    if not session or not llm_provider:
        return _error_return("Missing session or llm_provider in config")

    # Only process genuine cross-domain bridges, not intra-domain similarity edges.
    cross_domain_bridges = [
        b for b in bridges if b.get("source_domain") != b.get("target_domain")
    ]

    if not cross_domain_bridges:
        return {
            "hypotheses": [],
            "completed_steps": ["hypothesis_formulator"],
            "stream_events": [_stream_event(
                "step_complete", {"hypotheses_generated": 0, "reason": "no_cross_domain_bridges"}
            )],
        }

    loop = asyncio.get_event_loop()
    vs = VectorStore(session)
    hypotheses: list[Hypothesis] = []

    for bridge in cross_domain_bridges:
        # ── Fetch real chunk text for both endpoints ───────────────────────────
        # bridge["evidence_a/b"] may already be chunk text (from the updated
        # bridge_discoverer), but fall back to a fresh DB fetch if it looks like
        # a bare label (short, no spaces).
        ev_a = bridge.get("evidence_a", "")
        ev_b = bridge.get("evidence_b", "")

        if not ev_a or len(ev_a) < 80:
            ev_a = await _fetch_chunk_evidence(session, bridge["source_node_id"])
        if not ev_b or len(ev_b) < 80:
            ev_b = await _fetch_chunk_evidence(session, bridge["target_node_id"])

        # Derive display labels from the first line of evidence if possible,
        # otherwise use the domain names.
        source_label = ev_a.split(" [...] ")[0][:60].strip() or bridge["source_domain"]
        target_label = ev_b.split(" [...] ")[0][:60].strip() or bridge["target_domain"]

        prompt = _HYPOTHESIS_PROMPT.format(
            source_label=source_label,
            source_domain=bridge["source_domain"],
            target_label=target_label,
            target_domain=bridge["target_domain"],
            bridge_concept=bridge["bridge_concept"],
            evidence_a=ev_a[:700] or "(no evidence available)",
            evidence_b=ev_b[:700] or "(no evidence available)",
        )

        try:
            result = await loop.run_in_executor(
                None,
                lambda p=prompt: llm_provider.complete_json(p, max_tokens=600),
            )
        except Exception as exc:
            logger.warning(
                "hypothesis_formulator: LLM failed for bridge %s→%s — %s",
                bridge["source_node_id"], bridge["target_node_id"], exc,
            )
            continue

        statement = str(result.get("statement") or "").strip()
        if not statement or len(statement) < 40:
            logger.debug(
                "hypothesis_formulator: skipping empty/short statement for bridge %s→%s",
                bridge["source_domain"], bridge["target_domain"],
            )
            continue

        # Reject template output that slipped through
        banned_phrases = [
            "will yield measurable performance improvement",
            "will improve outcomes",
            "can enhance performance",
        ]
        if any(phrase in statement.lower() for phrase in banned_phrases):
            logger.info(
                "hypothesis_formulator: rejected template hypothesis for %s→%s",
                bridge["source_domain"], bridge["target_domain"],
            )
            continue

        null_hyp = str(result.get("null_hypothesis") or "").strip()
        hyp_type = str(result.get("hypothesis_type") or "technology_transfer")
        if hyp_type not in {"causal", "correlational", "technology_transfer", "mechanistic"}:
            hyp_type = "technology_transfer"
        method_hint = str(result.get("methodology_hint") or "").strip()
        key_pred = str(result.get("key_prediction") or "").strip()
        testability = min(max(float(result.get("testability_score", 0.6)), 0.0), 1.0)
        confidence = min(max(float(result.get("confidence", 0.6)), 0.0), 1.0)

        # ── Novelty self-check ─────────────────────────────────────────────────
        novelty = "uncertain"
        if embedder is not None:
            try:
                hyp_embedding = await loop.run_in_executor(
                    None,
                    lambda s=statement: embedder.embed(s),
                )
                similar = await vs.similarity_search(
                    hyp_embedding,
                    top_k=3,
                    min_score=_NOVELTY_THRESHOLD,
                )
                novelty = "known" if similar else "novel"
            except Exception as exc:
                logger.debug("hypothesis_formulator: novelty check failed — %s", exc)

        # Combine statement and key prediction for the hypothesis text stored in DB
        full_text = statement
        if key_pred and key_pred not in statement:
            full_text = f"{statement}\n\nKey prediction: {key_pred}"

        hyp: Hypothesis = {
            "bridge_edge_source": bridge["source_node_id"],
            "bridge_edge_target": bridge["target_node_id"],
            "statement": full_text,
            "null_hypothesis": null_hyp,
            "hypothesis_type": hyp_type,
            "methodology_hint": method_hint,
            "testability_score": testability,
            "novelty": novelty,
            "supporting_evidence": [ev_a[:300], ev_b[:300]],
            "confidence": confidence,
        }
        hypotheses.append(hyp)

        logger.info(
            "hypothesis_formulator: generated '%s' hypothesis (type=%s, novelty=%s, score=%.2f)",
            hyp_type, hyp_type, novelty, testability,
        )

        event = _stream_event(
            "hypothesis_generated",
            {
                "statement_preview": statement[:200],
                "novelty": novelty,
                "hypothesis_type": hyp_type,
                "bridge": f"{bridge['source_domain']} → {bridge['target_domain']}",
            },
        )

    event = _stream_event(
        "step_complete",
        {
            "hypotheses_generated": len(hypotheses),
            "bridges_processed": len(cross_domain_bridges),
        },
    )
    return {
        "hypotheses": hypotheses,
        "completed_steps": ["hypothesis_formulator"],
        "stream_events": [event],
    }


def _error_return(msg: str) -> dict:
    return {
        "hypotheses": [],
        "completed_steps": ["hypothesis_formulator"],
        "error": msg,
        "stream_events": [{
            "agent_node": "hypothesis_formulator",
            "event_type": "error",
            "content": {"error": msg},
            "timestamp": _now_iso(),
        }],
    }
