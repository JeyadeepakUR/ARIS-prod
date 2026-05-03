"""Server-Sent Events endpoint for real-time agent progress streaming.

GET /graphs/{graph_id}/stream

Emits agent_stream_events rows as they are written by the LangGraph worker.
The stream terminates automatically when the graph reaches a terminal status
(ready | failed) and all queued events have been sent.

Event format (each line):
    data: {"agent_node": "...", "event_type": "...", "content": {...}, "timestamp": "..."}

Special terminal event:
    data: {"event_type": "run_complete", "status": "ready" | "failed"}
"""
from __future__ import annotations

import asyncio
import json
import logging
from datetime import datetime, timezone
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, Request, status
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from sse_starlette.sse import EventSourceResponse

from apps.api.dependencies import get_current_user, get_db
from apps.api.models.graph import Graph
from apps.api.models.stream_event import AgentStreamEvent
from apps.api.models.user import User
from apps.api.models.workspace import Workspace

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/graphs/{graph_id}/stream", tags=["stream"])

# How long to wait between DB polls when no new events arrive (seconds)
_POLL_INTERVAL = 0.5
# Maximum time to keep the SSE connection open (seconds) — guards against zombies
_MAX_STREAM_SECONDS = 600


@router.get("")
async def stream_graph_events(
    graph_id: UUID,
    request: Request,
    session: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
) -> EventSourceResponse:
    """Stream real-time agent events for a graph build run."""

    # ── Authorise ─────────────────────────────────────────────────────────────
    graph = await session.scalar(select(Graph).where(Graph.id == graph_id))
    if graph is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Graph not found")

    workspace = await session.scalar(
        select(Workspace).where(
            Workspace.id == graph.workspace_id,
            Workspace.owner_id == current_user.id,
        )
    )
    if workspace is None:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Forbidden")

    # ── Build generator ───────────────────────────────────────────────────────
    async def event_generator():
        last_seen_at: datetime | None = None
        deadline = asyncio.get_event_loop().time() + _MAX_STREAM_SECONDS

        # Send a connection-confirmed ping immediately
        yield {
            "data": json.dumps({
                "event_type": "connected",
                "graph_id": str(graph_id),
                "timestamp": datetime.now(timezone.utc).isoformat(),
            })
        }

        while True:
            # Stop if client disconnected
            if await request.is_disconnected():
                logger.debug("SSE client disconnected for graph %s", graph_id)
                break

            # Stop if we've been running too long
            if asyncio.get_event_loop().time() > deadline:
                yield {
                    "data": json.dumps({
                        "event_type": "timeout",
                        "message": "Stream timeout — reconnect to resume.",
                    })
                }
                break

            # ── Fetch new events since last poll ──────────────────────────────
            try:
                q = (
                    select(AgentStreamEvent)
                    .where(AgentStreamEvent.graph_id == graph_id)
                    .order_by(AgentStreamEvent.created_at.asc())
                )
                if last_seen_at is not None:
                    q = q.where(AgentStreamEvent.created_at > last_seen_at)

                result = await session.execute(q)
                new_events = list(result.scalars())
            except Exception as exc:
                logger.error("SSE poll failed for graph %s: %s", graph_id, exc)
                await asyncio.sleep(_POLL_INTERVAL)
                continue

            for evt in new_events:
                yield {
                    "data": json.dumps({
                        "id": str(evt.id),
                        "agent_node": evt.agent_node,
                        "event_type": evt.event_type,
                        "content": evt.content,
                        "timestamp": evt.created_at.isoformat(),
                    })
                }
                last_seen_at = evt.created_at

                # Terminal event — stop streaming
                if evt.event_type in {"run_complete", "error"}:
                    await _cleanup_stream_events(session, graph_id)
                    return

            # ── Check graph terminal status (in case we missed the event) ─────
            try:
                current_graph = await session.scalar(
                    select(Graph).where(Graph.id == graph_id)
                )
            except Exception:
                current_graph = None

            if current_graph is not None and current_graph.status in {"ready", "failed"}:
                # Drain any remaining events not yet fetched
                if not new_events:
                    yield {
                        "data": json.dumps({
                            "event_type": "run_complete",
                            "status": current_graph.status,
                            "timestamp": datetime.now(timezone.utc).isoformat(),
                        })
                    }
                    await _cleanup_stream_events(session, graph_id)
                    break

            await asyncio.sleep(_POLL_INTERVAL)

    return EventSourceResponse(event_generator())


async def _cleanup_stream_events(session: AsyncSession, graph_id: UUID) -> None:
    """Delete all stream events for a graph once the run is complete."""
    try:
        events = await session.scalars(
            select(AgentStreamEvent).where(AgentStreamEvent.graph_id == graph_id)
        )
        for evt in events:
            await session.delete(evt)
        await session.commit()
    except Exception as exc:
        logger.warning("SSE cleanup failed for graph %s: %s", graph_id, exc)
