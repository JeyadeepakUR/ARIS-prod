"""Tests for SemanticRelationTool (Module 12)."""

from __future__ import annotations

import json
from typing import Any

import pytest

from aris.ml.relation_induction import RelationCandidate, SemanticRelationTool


def _make_input() -> dict[str, Any]:
    return {
        "text": "Temperature increase causes ice melting.",
        "entities": [
            {"span": "Temperature increase", "start": 0, "end": 20, "type": "PHENOMENON"},
            {"span": "ice melting", "start": 28, "end": 39, "type": "PHENOMENON"},
        ],
    }


def _patch_extract(tool: SemanticRelationTool, candidates: list[RelationCandidate]) -> None:
    def fake_extract(data: dict[str, Any]) -> list[RelationCandidate]:  # noqa: ARG001
        return candidates

    tool._extract_relations = fake_extract  # type: ignore[method-assign]


def test_determinism() -> None:
    tool = SemanticRelationTool()
    data = _make_input()
    input_json = json.dumps(data)

    candidate = RelationCandidate(
        relation_type="CAUSES",
        subject_span="Temperature increase",
        subject_start=0,
        subject_end=20,
        object_span="ice melting",
        object_start=28,
        object_end=39,
        predicate="cause",
        confidence=0.85,
        provenance={
            "model": tool.MODEL_NAME,
            "relation_ontology": list(tool.RELATION_ONTOLOGY),
            "strategy": "spacy_dependency_parse",
        },
    )

    _patch_extract(tool, [candidate])

    out1 = json.loads(tool.execute(input_json))
    out2 = json.loads(tool.execute(input_json))

    assert out1 == out2
    assert all(c["relation_type"] in tool.RELATION_ONTOLOGY for c in out1)
    assert all("relation_ontology" in c["provenance"] for c in out1)


def test_entity_reference_validity() -> None:
    tool = SemanticRelationTool()
    data = _make_input()
    input_json = json.dumps(data)

    candidate = RelationCandidate(
        relation_type="CAUSES",
        subject_span="Temperature increase",
        subject_start=0,
        subject_end=20,
        object_span="ice melting",
        object_start=28,
        object_end=39,
        predicate="cause",
        confidence=0.85,
        provenance={"model": "test"},
    )

    _patch_extract(tool, [candidate])

    output = json.loads(tool.execute(input_json))
    assert len(output) == 1
    rel = output[0]

    # Verify entity spans match input entities
    assert rel["subject_span"] == "Temperature increase"
    assert rel["subject_start"] == 0
    assert rel["subject_end"] == 20
    assert rel["object_span"] == "ice melting"
    assert rel["object_start"] == 28
    assert rel["object_end"] == 39


def test_confidence_bounds() -> None:
    tool = SemanticRelationTool()
    data = _make_input()

    # Test confidence clamping
    candidates = [
        RelationCandidate(
            relation_type="CAUSES",
            subject_span="A",
            subject_start=0,
            subject_end=1,
            object_span="B",
            object_start=2,
            object_end=3,
            predicate="cause",
            confidence=1.5,  # over bound
            provenance={},
        ),
        RelationCandidate(
            relation_type="USES",
            subject_span="C",
            subject_start=4,
            subject_end=5,
            object_span="D",
            object_start=6,
            object_end=7,
            predicate="use",
            confidence=-0.3,  # under bound
            provenance={},
        ),
    ]

    _patch_extract(tool, candidates)

    output = json.loads(tool.execute(json.dumps(data)))
    assert all(0.0 <= c["confidence"] <= 1.0 for c in output)


def test_failure_isolation() -> None:
    tool = SemanticRelationTool()
    data = _make_input()

    def raise_error(_: dict[str, Any]) -> list[RelationCandidate]:
        raise RuntimeError("SRL pipeline failed")

    tool._extract_relations = raise_error  # type: ignore[method-assign]

    output = json.loads(tool.execute(json.dumps(data)))
    assert "error" in output
    assert output.get("error_type") == "RuntimeError"
    assert output.get("model") == tool.MODEL_NAME
    assert "relation_ontology" in output


def test_closed_ontology_enforcement() -> None:
    tool = SemanticRelationTool()
    data = _make_input()

    # Candidate with valid relation type
    valid_candidate = RelationCandidate(
        relation_type="CAUSES",
        subject_span="A",
        subject_start=0,
        subject_end=1,
        object_span="B",
        object_start=2,
        object_end=3,
        predicate="cause",
        confidence=0.8,
        provenance={"relation_ontology": list(tool.RELATION_ONTOLOGY)},
    )

    _patch_extract(tool, [valid_candidate])

    output = json.loads(tool.execute(json.dumps(data)))
    assert len(output) == 1
    assert output[0]["relation_type"] in tool.RELATION_ONTOLOGY


def test_missing_spacy_dependency() -> None:
    from aris.ml.relation_induction import _MissingOptionalDependency

    tool = SemanticRelationTool()

    def raise_missing() -> None:
        raise _MissingOptionalDependency({"spacy"})

    tool._ensure_model = raise_missing  # type: ignore[method-assign]

    output = json.loads(tool.execute(json.dumps(_make_input())))
    assert output["error"] == "Missing optional dependency"
    assert output["module"] == "relation_induction"
    assert "spacy" in output["required"]
    assert "aris[ml]" in output["install_hint"]
