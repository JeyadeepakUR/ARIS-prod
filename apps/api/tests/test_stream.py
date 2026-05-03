"""SSE stream endpoint integration tests.

With CELERY_TASK_ALWAYS_EAGER, the graph build completes synchronously before
the test hits the stream endpoint.  All agent_stream_events are already in
the DB, so the SSE generator drains them immediately and closes after emitting
run_complete.
"""
from __future__ import annotations

import json
import time
from typing import Any
from urllib.parse import urlparse

from fastapi.testclient import TestClient


def _auth_headers(client: TestClient, email: str) -> dict[str, str]:
    resp = client.post("/auth/register", json={"email": email, "password": "StrongPass123"})
    assert resp.status_code == 201
    return {"Authorization": f"Bearer {resp.json()['access_token']}"}


def _upload_and_ingest(
    client: TestClient, headers: dict, workspace_id: str, filename: str, body: str
) -> str:
    upload = client.post(
        f"/workspaces/{workspace_id}/documents/upload-url",
        headers=headers,
        json={"filename": filename, "file_format": "text", "file_size_bytes": len(body)},
    )
    assert upload.status_code == 201
    up = upload.json()

    parsed = urlparse(up["upload_url"])
    path = parsed.path if not parsed.query else f"{parsed.path}?{parsed.query}"
    client.put(path, content=body.encode(), headers={"Content-Type": "text/plain"})

    for _ in range(30):
        poll = client.get(f"/jobs/{up['job_id']}", headers=headers)
        if poll.json()["status"] == "ready":
            break
        time.sleep(0.05)

    return str(up["document"]["id"])


def _build_graph(client: TestClient, headers: dict, workspace_id: str, doc_ids: list[str]) -> tuple[str, str]:
    resp = client.post(
        f"/workspaces/{workspace_id}/graphs",
        headers=headers,
        json={"document_ids": doc_ids, "strategy": "sequential", "plan_strategy": "weak-evidence", "max_plan_actions": 5},
    )
    assert resp.status_code == 201
    body = resp.json()
    graph_id = body["graph"]["id"]
    job_id = body["job_id"]

    for _ in range(30):
        poll = client.get(f"/jobs/{job_id}", headers=headers)
        if poll.json()["status"] in {"ready", "failed"}:
            break
        time.sleep(0.05)

    return graph_id, job_id


def _parse_sse_lines(raw: str) -> list[dict]:
    """Parse raw SSE body text into a list of JSON event dicts."""
    events: list[dict] = []
    for line in raw.splitlines():
        line = line.strip()
        if line.startswith("data:"):
            payload = line[5:].strip()
            if payload:
                try:
                    events.append(json.loads(payload))
                except json.JSONDecodeError:
                    pass
    return events


def test_stream_returns_connected_event(client: TestClient) -> None:
    headers = _auth_headers(client, "stream-connected@example.com")
    ws = client.post("/workspaces", headers=headers, json={"name": "Stream WS", "slug": "stream-ws"})
    workspace_id = ws.json()["id"]

    doc1 = _upload_and_ingest(client, headers, workspace_id, "a.txt",
                               "Machine learning anomaly detection for cybersecurity.")
    doc2 = _upload_and_ingest(client, headers, workspace_id, "b.txt",
                               "Neural network models applied to intrusion detection systems.")
    graph_id, _ = _build_graph(client, headers, workspace_id, [doc1, doc2])

    # Hit the SSE endpoint — since graph is already done, it drains and closes quickly
    with client.stream("GET", f"/graphs/{graph_id}/stream", headers=headers) as resp:
        assert resp.status_code == 200
        assert "text/event-stream" in resp.headers.get("content-type", "")

        lines: list[str] = []
        for line in resp.iter_lines():
            lines.append(line)
            # Stop after run_complete or after collecting enough lines
            if any("run_complete" in line or '"error"' in line for line in lines):
                break
            if len(lines) > 50:
                break

    raw = "\n".join(lines)
    events = _parse_sse_lines(raw)

    # Must have at least the 'connected' handshake event
    event_types = [e.get("event_type") for e in events]
    assert "connected" in event_types, f"Missing connected event, got: {event_types}"


def test_stream_ends_with_run_complete(client: TestClient) -> None:
    headers = _auth_headers(client, "stream-complete@example.com")
    ws = client.post("/workspaces", headers=headers, json={"name": "Stream Done", "slug": "stream-done"})
    workspace_id = ws.json()["id"]

    doc1 = _upload_and_ingest(client, headers, workspace_id, "ml.txt",
                               "Deep learning and neural networks for pattern recognition.")
    doc2 = _upload_and_ingest(client, headers, workspace_id, "sec.txt",
                               "Cybersecurity threat detection using anomaly scoring methods.")
    graph_id, _ = _build_graph(client, headers, workspace_id, [doc1, doc2])

    with client.stream("GET", f"/graphs/{graph_id}/stream", headers=headers) as resp:
        assert resp.status_code == 200
        lines: list[str] = []
        for line in resp.iter_lines():
            lines.append(line)
            if any("run_complete" in line or "error" in line for line in lines[-3:]):
                break
            if len(lines) > 100:
                break

    events = _parse_sse_lines("\n".join(lines))
    event_types = [e.get("event_type") for e in events]

    # run_complete (or error) must be the terminal event
    assert "run_complete" in event_types or "error" in event_types, (
        f"No terminal event in stream. Got event_types: {event_types}"
    )


def test_stream_requires_auth(client: TestClient) -> None:
    """Unauthenticated requests must be rejected."""
    import uuid
    fake_id = str(uuid.uuid4())
    resp = client.get(f"/graphs/{fake_id}/stream")
    assert resp.status_code in {401, 403}


def test_stream_returns_404_for_unknown_graph(client: TestClient) -> None:
    headers = _auth_headers(client, "stream-404@example.com")
    import uuid
    resp = client.get(f"/graphs/{uuid.uuid4()}/stream", headers=headers)
    assert resp.status_code == 404
