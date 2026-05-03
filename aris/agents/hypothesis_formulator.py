"""
Hypothesis Formulator agent node.

For each confirmed cross-domain bridge, generates a testable research
hypothesis. Performs a novelty self-check by searching the corpus for
semantically similar claims before assigning novelty status.

Dependencies injected through LangGraph config["configurable"]:
    session      — AsyncSession
    embedder     — Embedder (for novelty check)
    llm_provider — LLMProvider
"""
from __future__ import annotations

import asyncio
import logging
from datetime import datetime, timezone

from langchain_core.runnables import RunnableConfig

from aris.agents.state import BridgeEdge, Hypothesis, ResearchState, StreamEvent
from aris.retrieval.vector_store import VectorStore

logger = logging.getLogger(__name__)

_HYPOTHESIS_PROMPT = """\
A cross-domain knowledge bridge has been discovered in the research literature.

Source concept: "{source_concept}" (Domain: {source_domain})
Target concept: "{target_concept}" (Domain: {target_domain})
Bridge mechanism: {bridge_concept}
Evidence A: {evidence_a}
Evidence B: {evidence_b}

Generate one novel, testable research hypothesis that exploits this cross-domain \
transfer of knowledge.

Return ONLY a JSON object:
{{
  "statement": "The hypothesis (1-2 sentences, starts with 'We hypothesize that...')",
  "null_hypothesis": "The null hypothesis (1 sentence)",
  "hypothesis_type": "causal" | "correlational" | "technology_transfer" | "mechanistic",
  "methodology_hint": "Suggested experimental approach (1 sentence)",
  "testability_score": 0.0-1.0,
  "confidence": 0.0-1.0
}}"""

_NOVELTY_THRESHOLD = 0.85   # cosine score above which hypothesis is considered "known"


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _stream_event(event_type: str, content: dict) -> StreamEvent:
    return {
        "agent_node": "hypothesis_formulator",
        "event_type": event_type,
        "content": content,
        "timestamp": _now_iso(),
    }


async def hypothesis_formulator_node(state: ResearchState, config: RunnableConfig) -> dict:
    cfg = config.get("configurable", {})
    session = cfg.get("session")
    embedder = cfg.get("embedder")
    llm_provider = cfg.get("llm_provider")

    bridges: list[BridgeEdge] = state.get("bridges", [])

    if not session or not llm_provider:
        return _error_return("Missing session or llm_provider in config")

    if not bridges:
        return {
            "hypotheses": [],
            "completed_steps": ["hypothesis_formulator"],
            "stream_events": [_stream_event("step_complete", {"hypotheses_generated": 0, "reason": "no_bridges"})],
        }

    loop = asyncio.get_event_loop()
    vs = VectorStore(session)
    hypotheses: list[Hypothesis] = []

    for bridge in bridges:
        prompt = _HYPOTHESIS_PROMPT.format(
            source_concept=bridge["evidence_a"][:200],
            source_domain=bridge["source_domain"],
            target_concept=bridge["evidence_b"][:200],
            target_domain=bridge["target_domain"],
            bridge_concept=bridge["bridge_concept"],
            evidence_a=bridge["evidence_a"][:400],
            evidence_b=bridge["evidence_b"][:400],
        )

        try:
            result = await loop.run_in_executor(
                None,
                lambda p=prompt: llm_provider.complete_json(p, max_tokens=400),
            )
        except Exception as exc:
            logger.warning("hypothesis_formulator: LLM failed for bridge %s→%s — %s",
                           bridge["source_node_id"], bridge["target_node_id"], exc)
            continue

        statement = str(result.get("statement") or "").strip()
        if not statement:
            continue

        null_hyp = str(result.get("null_hypothesis") or "").strip()
        hyp_type = str(result.get("hypothesis_type") or "correlational")
        if hyp_type not in {"causal", "correlational", "technology_transfer", "mechanistic"}:
            hyp_type = "correlational"
        method_hint = str(result.get("methodology_hint") or "").strip()
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

        hyp: Hypothesis = {
            "bridge_edge_source": bridge["source_node_id"],
            "bridge_edge_target": bridge["target_node_id"],
            "statement": statement,
            "null_hypothesis": null_hyp,
            "hypothesis_type": hyp_type,
            "methodology_hint": method_hint,
            "testability_score": testability,
            "novelty": novelty,
            "supporting_evidence": [bridge["evidence_a"], bridge["evidence_b"]],
            "confidence": confidence,
        }
        hypotheses.append(hyp)

        event = _stream_event(
            "hypothesis_generated",
            {
                "statement": statement[:200],
                "novelty": novelty,
                "bridge": f"{bridge['source_domain']} → {bridge['target_domain']}",
            },
        )

    event = _stream_event(
        "step_complete",
        {"hypotheses_generated": len(hypotheses), "bridges_processed": len(bridges)},
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
