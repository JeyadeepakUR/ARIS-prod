"""Graph build task — invokes the LangGraph research agent network.

Uses graph.astream() (stream_mode="updates") so that stream_events emitted
by each agent are written to agent_stream_events after every step, making
them available to the SSE endpoint in near-real-time.
"""
from __future__ import annotations

import asyncio
import logging
from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime
from uuid import UUID, uuid4

from sqlalchemy import select

from apps.api.config import get_settings
from apps.api.core import database
from apps.api.models.document import Document
from apps.api.models.graph import Graph
from apps.api.models.job import AsyncJob
from apps.api.models.stream_event import AgentStreamEvent
from apps.worker.main import celery_app
from aris.agents.graph import build_research_graph
from aris.ingestion.embedder import Embedder
from aris.llm.provider import get_provider

logger = logging.getLogger(__name__)


@celery_app.task(name="apps.worker.tasks.graph_build_task.build_graph")
def build_graph(job_id: str) -> None:
    """Celery entry-point: run the async graph build in a dedicated event loop."""
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
        raise RuntimeError("Database session factory is not initialized")

    async with database.SessionLocal() as session:
        # ── Load job ──────────────────────────────────────────────────────────
        job = await session.scalar(select(AsyncJob).where(AsyncJob.id == UUID(job_id)))
        if job is None:
            await database.dispose_database()
            return

        payload = job.payload
        graph_id = UUID(str(payload.get("graph_id")))
        workspace_id = UUID(str(payload.get("workspace_id")))
        document_ids = [str(v) for v in payload.get("document_ids", [])]

        graph = await session.scalar(
            select(Graph).where(Graph.id == graph_id, Graph.workspace_id == workspace_id)
        )
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
        # Capture metadata_json as a plain dict now — after commit() the ORM
        # object is expired and reading it later triggers a sync lazy-load that
        # fails with MissingGreenlet in an async session.
        existing_graph_meta = dict(graph.metadata_json or {})
        await session.commit()

        try:
            # ── Validate documents ────────────────────────────────────────────
            rows = await session.scalars(
                select(Document).where(
                    Document.workspace_id == workspace_id,
                    Document.id.in_([UUID(d) for d in document_ids]),
                )
            )
            ordered_docs = list(rows)
            if len(ordered_docs) != len(document_ids):
                raise ValueError("One or more documents do not belong to workspace")
            if any(doc.status != "ready" for doc in ordered_docs):
                raise ValueError("All documents must be in 'ready' status before graph build")

            # ── Infrastructure ────────────────────────────────────────────────
            embedder = Embedder(
                model=settings.embed_model,
                base_url=settings.embed_base_url or settings.ollama_base_url,
            )
            try:
                llm_provider = get_provider(
                    settings.llm_provider,
                    model=settings.llm_model,
                    api_key=settings.llm_api_key,
                    base_url=settings.ollama_base_url,
                )
            except Exception:
                from aris.llm.mock_provider import MockProvider
                llm_provider = MockProvider()
                logger.warning("graph_build_task: LLM provider unavailable, using MockProvider")

            thread_id = str(uuid4())
            initial_state = {
                "graph_id": str(graph_id),
                "workspace_id": str(workspace_id),
                "document_ids": document_ids,
                "thread_id": thread_id,
                "concepts": [],
                "intra_edges": [],
                "bridge_candidates": [],
                "bridges": [],
                "contradictions": [],
                "hypotheses": [],
                "gaps": [],
                "stream_events": [],
                "orchestrator_plan": [
                    "concept_extractor",
                    "bridge_discoverer",
                    "contradiction_analyst",
                    "hypothesis_formulator",
                    "gap_analyst",
                ],
                "completed_steps": [],
                "iteration": 0,
                "error": None,
                "human_feedback": None,
                "_next_node": "orchestrator",
            }

            langgraph_config = {
                "configurable": {
                    "thread_id": thread_id,
                    "session": session,
                    "embedder": embedder,
                    "llm_provider": llm_provider,
                },
                "recursion_limit": 30,
            }

            # ── Run LangGraph, capturing stream events after each step ────────
            compiled_graph = build_research_graph()
            final_state: dict = {}

            async for chunk in compiled_graph.astream(
                initial_state,
                config=langgraph_config,
                stream_mode="updates",
            ):
                # chunk: {node_name: state_delta_dict}
                for node_name, delta in chunk.items():
                    new_events = delta.get("stream_events", [])
                    if new_events:
                        await _persist_stream_events(
                            session, graph_id, thread_id, node_name, new_events
                        )
                    # Merge delta into final_state (accumulate)
                    for key, value in delta.items():
                        if isinstance(value, list) and isinstance(final_state.get(key), list):
                            final_state[key] = final_state[key] + value
                        else:
                            final_state[key] = value

                # Commit after every step so SSE endpoint sees events immediately
                await session.commit()

                # Diagnostic: log node count after each step commit (portable
                # across PG and SQLite — uses ORM filter rather than raw SQL).
                if node_name == "concept_extractor":
                    from sqlalchemy import func as _func, select as _sa_select
                    from apps.api.models.node import Node as _NodeModel
                    _r = await session.execute(
                        _sa_select(_func.count(_NodeModel.id)).where(
                            _NodeModel.graph_id == graph_id
                        )
                    )
                    logger.info(
                        "graph_build_task: nodes in DB after concept_extractor commit = %d",
                        _r.scalar() or 0,
                    )

            # ── Update graph + job from final accumulated state ────────────────
            concepts = final_state.get("concepts", [])
            bridges = final_state.get("bridges", [])
            contradictions = final_state.get("contradictions", [])
            hypotheses = final_state.get("hypotheses", [])
            gaps = final_state.get("gaps", [])
            agent_error = final_state.get("error")

            graph.status = "ready"
            graph.metadata_json = {
                **existing_graph_meta,
                "concepts_count": len(concepts),
                "bridges_count": len(bridges),
                "contradictions_count": len(contradictions),
                "hypotheses_count": len(hypotheses),
                "gaps_count": len(gaps),
                "thread_id": thread_id,
            }

            job.status = "ready"
            job.error = agent_error
            job.result = {
                "graph_id": str(graph_id),
                "concepts_count": len(concepts),
                "bridges_count": len(bridges),
                "contradictions_count": len(contradictions),
                "hypotheses_count": len(hypotheses),
                "gaps_count": len(gaps),
            }
            job.trace_json = {
                "thread_id": thread_id,
                "stream_events_count": len(final_state.get("stream_events", [])),
            }

            # Emit terminal run_complete event
            session.add(
                AgentStreamEvent(
                    id=uuid4(),
                    graph_id=graph_id,
                    thread_id=thread_id,
                    agent_node="system",
                    event_type="run_complete",
                    content={
                        "status": "ready",
                        "concepts_count": len(concepts),
                        "bridges_count": len(bridges),
                        "contradictions_count": len(contradictions),
                        "hypotheses_count": len(hypotheses),
                        "gaps_count": len(gaps),
                    },
                )
            )

        except Exception as exc:
            logger.exception("graph_build_task: failed for job %s", job_id)
            graph.status = "failed"
            job.status = "failed"
            job.error = str(exc)
            job.result = {"graph_id": str(graph_id), "status": "failed"}
            job.trace_json = {"error": str(exc)}

            # Emit error event so SSE clients know the run failed
            session.add(
                AgentStreamEvent(
                    id=uuid4(),
                    graph_id=graph_id,
                    thread_id=str(thread_id) if "thread_id" in dir() else "unknown",
                    agent_node="system",
                    event_type="error",
                    content={"error": str(exc)},
                )
            )

        finally:
            job.completed_at = datetime.now(UTC)
            await session.commit()

            if job.status == "ready":
                try:
                    from apps.api.tasks.hypothesis_task import generate_hypotheses
                    generate_hypotheses.si(str(graph_id)).delay()
                except Exception as exc:
                    logger.warning("graph_build_task: could not queue hypothesis task — %s", exc)

    await database.dispose_database()


async def _persist_stream_events(
    session,
    graph_id: UUID,
    thread_id: str,
    node_name: str,
    events: list[dict],
) -> None:
    """Write a batch of StreamEvent dicts to the agent_stream_events table."""
    for evt in events:
        session.add(
            AgentStreamEvent(
                id=uuid4(),
                graph_id=graph_id,
                thread_id=thread_id,
                agent_node=evt.get("agent_node", node_name),
                event_type=evt.get("event_type", "unknown"),
                content=evt.get("content", {}),
            )
        )
