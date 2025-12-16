"""Tests for OntologyClusteringTool (Module 11)."""

from __future__ import annotations

import json
from typing import Callable, List

import pytest

from aris.ml.ontology_induction import (
    OntologyClusterCandidate,
    OntologyClusteringTool,
)


def _make_items() -> list[dict[str, str]]:
    return [
        {"id": "a", "text": "Cell biology study of membranes"},
        {"id": "b", "text": "Membrane transport experiment"},
        {"id": "c", "text": "Quantum field theory introduction"},
        {"id": "d", "text": "Advanced topics in QFT"},
        {"id": "e", "text": "Unrelated noise sentence"},
    ]


def _patch_embed(tool: OntologyClusteringTool, vectors: list[list[float]]) -> None:
    def fake_embed(texts: list[str]) -> list[list[float]]:  # noqa: ARG001
        return vectors

    tool._embed = fake_embed  # type: ignore[attr-defined]


def _patch_hdbscan(tool: OntologyClusteringTool, labels: list[int], probs: list[float], persistence: dict[int, float]) -> None:
    def fake_cluster(embeddings: list[list[float]]):  # noqa: ARG001
        return labels, probs, persistence

    tool._hdbscan_cluster = fake_cluster  # type: ignore[attr-defined]


def test_determinism() -> None:
    tool = OntologyClusteringTool()
    items = _make_items()
    # Two clear clusters: [a,b], [c,d], with e as noise
    vectors = [
        [1.0, 0.0],  # a
        [0.9, 0.1],  # b
        [0.0, 1.0],  # c
        [0.1, 0.9],  # d
        [5.0, 5.0],  # e noise
    ]
    # Patch out clustering deps check
    tool._ensure_clustering_deps = lambda: None  # type: ignore[method-assign]
    _patch_embed(tool, vectors)
    _patch_hdbscan(
        tool,
        labels=[0, 0, 1, 1, -1],
        probs=[0.9, 0.8, 0.85, 0.82, 0.1],
        persistence={0: 0.77, 1: 0.73},
    )

    input_json = json.dumps(items)
    out1 = json.loads(tool.execute(input_json))
    out2 = json.loads(tool.execute(input_json))

    assert out1 == out2
    assert all("hdbscan_params" in c["provenance"] for c in out1)
    assert all(c["provenance"]["model"] == tool.MODEL_NAME for c in out1)


def test_input_immutability() -> None:
    tool = OntologyClusteringTool()
    items = _make_items()
    frozen = json.dumps(items)
    _patch_embed(tool, [[0.0, 0.0] for _ in items])
    _patch_hdbscan(tool, labels=[-1] * len(items), probs=[1.0] * len(items), persistence={})

    before = json.loads(frozen)
    _ = tool.execute(frozen)
    after = json.loads(frozen)
    assert before == after


def test_cluster_integrity() -> None:
    tool = OntologyClusteringTool()
    items = _make_items()
    tool._ensure_clustering_deps = lambda: None  # type: ignore[method-assign]
    _patch_embed(tool, [[0.0, 0.0] for _ in items])
    _patch_hdbscan(tool, labels=[0, 0, 1, 1, -1], probs=[1.0] * len(items), persistence={0: 0.5, 1: 0.6})

    output = json.loads(tool.execute(json.dumps(items)))
    all_members = [m for c in output for m in c["member_ids"]]
    assert sorted(all_members) == ["a", "b", "c", "d"]
    assert len(set(all_members)) == 4  # no duplicates
    for c in output:
        assert 0.0 <= c["stability"] <= 1.0
        assert c["size"] == len(c["member_ids"])


def test_failure_isolation() -> None:
    tool = OntologyClusteringTool()
    items = _make_items()

    # Patch out _ensure_clustering_deps so we skip that check
    def no_op() -> None:
        pass

    tool._ensure_clustering_deps = no_op  # type: ignore[method-assign]

    def bad_embed(_: list[str]) -> list[list[float]]:
        raise RuntimeError("GPU down")

    tool._embed = bad_embed  # type: ignore[attr-defined]

    output = json.loads(tool.execute(json.dumps(items)))
    assert "error" in output
    assert output.get("error_type") == "RuntimeError"
    assert output.get("model") == tool.MODEL_NAME
