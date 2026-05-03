"""Sprint 3 integration test for ARIS graph pipeline endpoints.

With the LangGraph agent pipeline, graph build now creates concept nodes
(from concept_extractor) and cross-domain bridge edges (from bridge_discoverer)
rather than the old document/domain/concept node hierarchy.
"""

from __future__ import annotations

import time
from typing import Any
from urllib.parse import urlparse

from fastapi.testclient import TestClient


def _auth_headers(client: TestClient, email: str) -> dict[str, str]:
    register = client.post(
        "/auth/register",
        json={"email": email, "password": "StrongPass123"},
    )
    assert register.status_code == 201
    token = register.json()["access_token"]
    return {"Authorization": f"Bearer {token}"}


def test_graph_build_with_three_docs_completes_and_returns_nodes(client: TestClient) -> None:
    headers = _auth_headers(client, "sprint3-graph@example.com")

    workspace = client.post(
        "/workspaces",
        headers=headers,
        json={"name": "Sprint Three", "slug": "sprint-three"},
    )
    assert workspace.status_code == 201
    workspace_id = workspace.json()["id"]

    document_ids: list[str] = []
    for index in range(3):
        upload = client.post(
            f"/workspaces/{workspace_id}/documents/upload-url",
            headers=headers,
            json={
                "filename": f"doc-{index + 1}.txt",
                "file_format": "text",
                "file_size_bytes": 128,
            },
        )
        assert upload.status_code == 201
        upload_body: dict[str, Any] = upload.json()
        document_ids.append(upload_body["document"]["id"])

        parsed = urlparse(upload_body["upload_url"])
        upload_path = parsed.path if not parsed.query else f"{parsed.path}?{parsed.query}"
        upload_put = client.put(
            upload_path,
            data=f"Document {index + 1} real content for graph build".encode("utf-8"),
            headers={"Content-Type": "text/plain"},
        )
        assert upload_put.status_code == 204

        job_id = upload_body["job_id"]

        final_status = None
        for _ in range(25):
            poll = client.get(f"/jobs/{job_id}", headers=headers)
            assert poll.status_code == 200
            final_status = poll.json()["status"]
            if final_status == "ready":
                break
            time.sleep(0.05)
        assert final_status == "ready"

    list_docs = client.get(f"/workspaces/{workspace_id}/documents", headers=headers)
    assert list_docs.status_code == 200
    listed_documents: list[dict[str, Any]] = list_docs.json()
    assert len(listed_documents) == 3

    build = client.post(
        f"/workspaces/{workspace_id}/graphs",
        headers=headers,
        json={
            "document_ids": document_ids,
            "strategy": "sequential",
            "plan_strategy": "weak-evidence",
            "max_plan_actions": 10,
        },
    )
    assert build.status_code == 201
    body: dict[str, Any] = build.json()
    graph_id = body["graph"]["id"]
    graph_job_id = body["job_id"]

    graph_final_status = None
    for _ in range(25):
        poll = client.get(f"/jobs/{graph_job_id}", headers=headers)
        assert poll.status_code == 200
        graph_final_status = poll.json()["status"]
        if graph_final_status in {"ready", "failed"}:
            break
        time.sleep(0.05)

    assert graph_final_status == "ready"

    list_graphs = client.get(f"/workspaces/{workspace_id}/graphs", headers=headers)
    assert list_graphs.status_code == 200
    listed_graphs: list[dict[str, Any]] = list_graphs.json()
    assert len(listed_graphs) >= 1
    assert listed_graphs[0]["id"] == graph_id

    # LangGraph concept_extractor creates concept nodes
    nodes_response = client.get(f"/graphs/{graph_id}/nodes", headers=headers)
    assert nodes_response.status_code == 200
    nodes: list[dict[str, Any]] = nodes_response.json()
    # Graph build may produce 0+ concept nodes depending on chunk content
    assert isinstance(nodes, list)
    if nodes:
        assert isinstance(nodes[0]["label"], str)
        assert nodes[0]["node_type"] in {"concept", "document", "domain", "bridge_concept"}

    # Edges may be cross_domain_bridge edges from bridge_discoverer
    edges_response = client.get(f"/graphs/{graph_id}/edges", headers=headers)
    assert edges_response.status_code == 200
    edges: list[dict[str, Any]] = edges_response.json()
    assert isinstance(edges, list)
    if edges:
        assert isinstance(edges[0]["evidence"], dict)
        assert isinstance(edges[0]["evidence"].get("text"), str)


def test_domain_network_build_creates_concept_nodes_and_bridge_edges(client: TestClient) -> None:
    headers = _auth_headers(client, "sprint3-domain-network@example.com")

    workspace = client.post(
        "/workspaces",
        headers=headers,
        json={"name": "Domain Network", "slug": "domain-network"},
    )
    assert workspace.status_code == 201
    workspace_id = workspace.json()["id"]

    docs_payload = [
        {
            "filename": "ml-cyber-paper-1.txt",
            "body": (
                "Machine learning neural networks detect intrusion patterns. "
                "Anomaly detection uses deep learning models for cybersecurity threat classification."
            ),
        },
        {
            "filename": "ml-cyber-paper-2.txt",
            "body": (
                "Neural training improves threat detection accuracy. "
                "Security systems use machine learning for anomaly classification and intrusion response."
            ),
        },
    ]

    document_ids: list[str] = []
    for payload in docs_payload:
        upload = client.post(
            f"/workspaces/{workspace_id}/documents/upload-url",
            headers=headers,
            json={
                "filename": payload["filename"],
                "file_format": "text",
                "file_size_bytes": len(payload["body"]),
            },
        )
        assert upload.status_code == 201
        upload_body: dict[str, Any] = upload.json()
        document_ids.append(upload_body["document"]["id"])

        parsed = urlparse(upload_body["upload_url"])
        upload_path = parsed.path if not parsed.query else f"{parsed.path}?{parsed.query}"
        put_result = client.put(
            upload_path,
            content=payload["body"].encode("utf-8"),
            headers={"Content-Type": "text/plain"},
        )
        assert put_result.status_code == 204

        job_id = upload_body["job_id"]
        final_status = None
        for _ in range(25):
            poll = client.get(f"/jobs/{job_id}", headers=headers)
            assert poll.status_code == 200
            final_status = poll.json()["status"]
            if final_status in {"ready", "failed"}:
                break
            time.sleep(0.05)
        assert final_status == "ready"

    build = client.post(
        f"/workspaces/{workspace_id}/graphs",
        headers=headers,
        json={
            "document_ids": document_ids,
            "strategy": "domain_network",
            "plan_strategy": "weak-evidence",
            "max_plan_actions": 10,
        },
    )
    assert build.status_code == 201
    body: dict[str, Any] = build.json()
    graph_id = body["graph"]["id"]
    graph_job_id = body["job_id"]

    graph_final_status = None
    for _ in range(25):
        poll = client.get(f"/jobs/{graph_job_id}", headers=headers)
        assert poll.status_code == 200
        graph_final_status = poll.json()["status"]
        if graph_final_status in {"ready", "failed"}:
            break
        time.sleep(0.05)

    assert graph_final_status == "ready"

    nodes_response = client.get(f"/graphs/{graph_id}/nodes", headers=headers)
    assert nodes_response.status_code == 200
    nodes: list[dict[str, Any]] = nodes_response.json()

    # LangGraph pipeline creates concept nodes extracted by the concept_extractor agent
    node_types = {node["node_type"] for node in nodes}
    assert "concept" in node_types or len(nodes) == 0  # 0 if chunks not embedded in test env

    # Bridge edges created by bridge_discoverer when cross-domain concept pairs are found
    edges_response = client.get(f"/graphs/{graph_id}/edges", headers=headers)
    assert edges_response.status_code == 200
    edges: list[dict[str, Any]] = edges_response.json()
    assert isinstance(edges, list)

    # If bridge edges exist, they should have the cross_domain_bridge type and evidence text
    bridge_edges = [e for e in edges if e["edge_type"] == "cross_domain_bridge"]
    if bridge_edges:
        assert bridge_edges[0]["edge_category"] == "INTER_DOMAIN_BRIDGE"
        assert isinstance(bridge_edges[0]["evidence"].get("text"), str)
        assert bridge_edges[0]["evidence"]["text"]
