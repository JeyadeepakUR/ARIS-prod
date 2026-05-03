"""
Gap Analyst agent node.

Finds node pairs with ≥2 common graph neighbours but no direct edge —
structural holes in the knowledge graph that represent research gaps.
Asks the LLM to describe each gap and why it is worth investigating.
Persists findings as PlanAction rows.

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

from apps.api.models.plan import PlanAction
from aris.agents.state import ResearchGap, ResearchState, StreamEvent
from aris.retrieval.vector_store import VectorStore

logger = logging.getLogger(__name__)

_GAP_PROMPT = """\
In a knowledge graph built from research literature, two concepts share \
{shared_count} common connections but have no direct link established \
between them.

Concept A: "{label_a}"
Concept B: "{label_b}"
Number of shared connections: {shared_count}

Describe this research gap: why is there no direct study of the relationship \
between A and B, and what scientific value would bridging this gap provide?

Return ONLY a JSON object:
{{
  "gap_description": "What is missing and why it matters (2-3 sentences)",
  "investigation_priority": 0.0-1.0,
  "rationale": "Which disciplines should investigate this and what approach (1-2 sentences)"
}}"""


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _stream_event(event_type: str, content: dict) -> StreamEvent:
    return {
        "agent_node": "gap_analyst",
        "event_type": event_type,
        "content": content,
        "timestamp": _now_iso(),
    }


async def gap_analyst_node(state: ResearchState, config: RunnableConfig) -> dict:
    cfg = config.get("configurable", {})
    session = cfg.get("session")
    llm_provider = cfg.get("llm_provider")

    graph_id: str = state["graph_id"]

    if not session or not llm_provider:
        return _error_return("Missing session or llm_provider in config")

    # ── 1. Find structural gaps via pgvector helper ───────────────────────────
    vs = VectorStore(session)
    try:
        gap_rows = await vs.find_graph_gaps(
            graph_id=graph_id,
            min_shared=2,
            limit=20,
        )
    except Exception as exc:
        logger.error("gap_analyst: gap query failed — %s", exc)
        return _error_return(str(exc))

    if not gap_rows:
        return {
            "gaps": [],
            "completed_steps": ["gap_analyst"],
            "stream_events": [_stream_event("step_complete", {"gaps_found": 0, "reason": "no_structural_gaps"})],
        }

    # ── 2. LLM description for each gap ──────────────────────────────────────
    loop = asyncio.get_event_loop()
    gaps: list[ResearchGap] = []
    plan_actions: list[PlanAction] = []

    for row in gap_rows:
        prompt = _GAP_PROMPT.format(
            label_a=row["n1_label"],
            label_b=row["n2_label"],
            shared_count=row["shared_count"],
        )
        try:
            result = await loop.run_in_executor(
                None,
                lambda p=prompt: llm_provider.complete_json(p, max_tokens=350),
            )
        except Exception as exc:
            logger.warning("gap_analyst: LLM failed — %s", exc)
            continue

        gap_desc = str(result.get("gap_description") or "").strip()
        priority = min(max(float(result.get("investigation_priority", 0.5)), 0.0), 1.0)
        rationale = str(result.get("rationale") or "").strip()

        if not gap_desc:
            continue

        gap: ResearchGap = {
            "node_a_id": row["n1_id"],
            "node_b_id": row["n2_id"],
            "common_neighbor_ids": [],    # row doesn't carry ids, only count
            "gap_description": gap_desc,
            "investigation_priority": priority,
            "rationale": rationale,
        }
        gaps.append(gap)

        # Persist as PlanAction
        action = PlanAction(
            id=uuid4(),
            graph_id=UUID(graph_id),
            action_type="investigate_gap",
            description=f"Bridge research gap between '{row['n1_label']}' and '{row['n2_label']}'",
            evidence=f"Shared {row['shared_count']} common connections but no direct edge.",
            rationale=f"{gap_desc}\n\n{rationale}",
            priority=priority,
            status="pending",
            metadata_json={
                "node_a_id": row["n1_id"],
                "node_b_id": row["n2_id"],
                "shared_count": int(row["shared_count"]),
            },
        )
        session.add(action)
        plan_actions.append(action)

        _stream_event(
            "gap_identified",
            {
                "node_a": row["n1_label"],
                "node_b": row["n2_label"],
                "priority": priority,
            },
        )

    await session.flush()

    event = _stream_event(
        "step_complete",
        {"gaps_found": len(gaps), "plan_actions_written": len(plan_actions)},
    )
    return {
        "gaps": gaps,
        "completed_steps": ["gap_analyst"],
        "stream_events": [event],
    }


def _error_return(msg: str) -> dict:
    return {
        "gaps": [],
        "completed_steps": ["gap_analyst"],
        "error": msg,
        "stream_events": [{
            "agent_node": "gap_analyst",
            "event_type": "error",
            "content": {"error": msg},
            "timestamp": _now_iso(),
        }],
    }
