"""
Contradiction Analyst agent node.

Finds semantically similar chunk pairs across different documents, then
asks the LLM to determine whether they make contradictory claims.
Persists confirmed contradictions to the contradictions table.

Dependencies injected through LangGraph config["configurable"]:
    session      — AsyncSession
    llm_provider — LLMProvider
"""
from __future__ import annotations

import asyncio
import logging
from datetime import UTC, datetime
from uuid import uuid4

from langchain_core.runnables import RunnableConfig

from aris.agents.state import Contradiction, ResearchState, StreamEvent
from aris.retrieval.vector_store import VectorStore

logger = logging.getLogger(__name__)

_CONTRADICTION_PROMPT = """\
Two research text excerpts from different documents discuss related topics. \
Determine whether they make contradictory claims.

Excerpt A:
{content_a}

Excerpt B:
{content_b}

Return ONLY a JSON object:
{{
  "contradicts": true/false,
  "contradiction_type": "direct" | "methodological" | "scope" | null,
  "severity": 0.0-1.0,
  "reasoning": "one-sentence explanation",
  "claim_a_summary": "core claim from excerpt A (max 100 chars)",
  "claim_b_summary": "core claim from excerpt B (max 100 chars)"
}}

contradiction_type guide:
- direct: same topic, opposite conclusions
- methodological: same topic, different methods, different outcomes
- scope: one claim is a subset/superset that conflicts with the other"""

_SIMILAR_THRESHOLD = 0.80


def _now_iso() -> str:
    return datetime.now(UTC).isoformat()


def _stream_event(event_type: str, content: dict) -> StreamEvent:
    return {
        "agent_node": "contradiction_analyst",
        "event_type": event_type,
        "content": content,
        "timestamp": _now_iso(),
    }


async def contradiction_analyst_node(state: ResearchState, config: RunnableConfig) -> dict:
    cfg = config.get("configurable", {})
    session = cfg.get("session")
    llm_provider = cfg.get("llm_provider")

    graph_id: str = state["graph_id"]
    document_ids: list[str] = state["document_ids"]

    if not session or not llm_provider:
        return _error_return("Missing session or llm_provider in config")

    # ── 1. Find similar cross-document chunk pairs via pgvector ───────────────
    vs = VectorStore(session)
    try:
        # Cap to 12 pairs to stay friendly with free-tier LLM rate limits.
        pairs = await vs.cross_document_similar_chunks(
            document_ids=document_ids,
            min_score=_SIMILAR_THRESHOLD,
            limit=12,
        )
    except Exception as exc:
        logger.error("contradiction_analyst: chunk search failed — %s", exc)
        return _error_return(str(exc))

    if not pairs:
        return {
            "contradictions": [],
            "completed_steps": ["contradiction_analyst"],
            "stream_events": [
                _stream_event(
                    "step_complete",
                    {"contradictions_found": 0, "reason": "no_similar_pairs"},
                )
            ],
        }

    # ── 2. Fetch document author metadata (portable across PG / SQLite) ───────
    loop = asyncio.get_event_loop()
    doc_authors: dict[str, str] = {}
    try:
        from uuid import UUID

        from sqlalchemy import select as sa_select

        from apps.api.models.document import Document
        rows = await session.execute(
            sa_select(Document.id, Document.metadata_json)
            .where(Document.id.in_([UUID(d) for d in document_ids]))
        )
        for r in rows.fetchall():
            meta = r.metadata_json or {}
            authors = meta.get("authors") or "Unknown"
            if isinstance(authors, list):
                authors = ", ".join(str(a) for a in authors) or "Unknown"
            doc_authors[str(r.id)] = str(authors)
    except Exception:
        pass

    # ── 3. LLM contradiction check for each similar pair ──────────────────────
    found_contradictions: list[Contradiction] = []
    checked: set[frozenset[str]] = set()

    for row in pairs:
        pair_key = frozenset({row["chunk1_id"], row["chunk2_id"]})
        if pair_key in checked:
            continue
        checked.add(pair_key)

        prompt = _CONTRADICTION_PROMPT.format(
            content_a=row["content1"][:800],
            content_b=row["content2"][:800],
        )
        try:
            result = await loop.run_in_executor(
                None,
                lambda p=prompt: llm_provider.complete_json(p, max_tokens=350),
            )
        except Exception as exc:
            logger.warning("contradiction_analyst: LLM call failed — %s", exc)
            continue

        if not result.get("contradicts"):
            continue

        author_a = doc_authors.get(row["doc1_id"], "Unknown")
        author_b = doc_authors.get(row["doc2_id"], "Unknown")
        claim_a = str(result.get("claim_a_summary") or row["content1"][:200]).strip()
        claim_b = str(result.get("claim_b_summary") or row["content2"][:200]).strip()
        ctype = str(result.get("contradiction_type") or "direct")
        if ctype not in {"direct", "methodological", "scope"}:
            ctype = "direct"
        severity = min(max(float(result.get("severity", 0.5)), 0.0), 1.0)
        reasoning = str(result.get("reasoning") or "").strip()

        contradiction: Contradiction = {
            "claim_a_text": claim_a,
            "claim_b_text": claim_b,
            "claim_a_chunk_id": row["chunk1_id"],
            "claim_b_chunk_id": row["chunk2_id"],
            "author_a": author_a,
            "author_b": author_b,
            "contradiction_type": ctype,
            "severity": severity,
            "llm_reasoning": reasoning,
        }
        found_contradictions.append(contradiction)
        await _persist_contradiction(session, graph_id, contradiction)

    await session.flush()

    event = _stream_event(
        "step_complete",
        {"contradictions_found": len(found_contradictions), "pairs_evaluated": len(pairs)},
    )
    return {
        "contradictions": found_contradictions,
        "completed_steps": ["contradiction_analyst"],
        "stream_events": [event],
    }


async def _persist_contradiction(session, graph_id: str, c: Contradiction) -> None:
    """Insert a contradiction row using the ORM model (portable across PG/SQLite)."""
    from uuid import UUID

    from apps.api.models.contradiction import Contradiction as ContradictionModel
    session.add(
        ContradictionModel(
            id=uuid4(),
            graph_id=UUID(graph_id),
            claim_a_chunk_id=UUID(c["claim_a_chunk_id"]),
            claim_b_chunk_id=UUID(c["claim_b_chunk_id"]),
            author_a=c["author_a"],
            author_b=c["author_b"],
            claim_a_text=c["claim_a_text"][:2000],
            claim_b_text=c["claim_b_text"][:2000],
            contradiction_type=c["contradiction_type"],
            severity=c["severity"],
            llm_reasoning=c["llm_reasoning"][:2000],
            status="open",
        )
    )


def _error_return(msg: str) -> dict:
    return {
        "contradictions": [],
        "completed_steps": ["contradiction_analyst"],
        "error": msg,
        "stream_events": [{
            "agent_node": "contradiction_analyst",
            "event_type": "error",
            "content": {"error": msg},
            "timestamp": _now_iso(),
        }],
    }
