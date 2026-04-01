"""Tests for Hypothesis Impact Scoring (Module 13 Enhancement)."""

from __future__ import annotations

import json
from typing import Any

import pytest

from aris.graph.hypothesis_impact import HypothesisImpactScore, HypothesisImpactScoringTool


def _make_input() -> dict[str, Any]:
    """Sample input with hypotheses, entities, and relations."""
    return {
        "hypotheses": [
            {
                "hypothesis_id": "hyp-1",
                "hypothesis_type": "gap",
                "description": "Nodes 'BERT' and 'GPT' share embedding similarities but are not directly connected",
                "confidence": 0.8,
                "supporting_nodes": ["node-1", "node-2"],
                "supporting_edges": [],
            },
            {
                "hypothesis_id": "hyp-2",
                "hypothesis_type": "contradiction",
                "description": "Contradictory relationships between 'BERT' and 'RoBERTa': supports vs contradicts",
                "confidence": 0.95,
                "supporting_nodes": ["node-3", "node-4"],
                "supporting_edges": ["edge-1", "edge-2"],
            },
            {
                "hypothesis_id": "hyp-3",
                "hypothesis_type": "hub",
                "description": "Node 'BERT' is a hub with many connections to datasets and benchmarks",
                "confidence": 0.7,
                "supporting_nodes": ["node-5"],
                "supporting_edges": ["edge-3", "edge-4", "edge-5"],
            },
        ],
        "entity_mentions": {
            "BERT": {"mention_count": 42, "entity_type": "METHOD"},
            "GPT": {"mention_count": 38, "entity_type": "METHOD"},
            "RoBERTa": {"mention_count": 15, "entity_type": "METHOD"},
            "GLUE": {"mention_count": 60, "entity_type": "DATASET"},
            "92.8%": {"mention_count": 2, "entity_type": "METRIC"},
        },
        "relations": [
            {
                "relation_type": "OUTPERFORMS",
                "subject_span": "BERT",
                "object_span": "GPT",
                "confidence": 0.85,
            },
            {
                "relation_type": "ACHIEVES",
                "subject_span": "BERT",
                "object_span": "92.8%",
                "confidence": 0.8,
            },
            {
                "relation_type": "USES",
                "subject_span": "BERT",
                "object_span": "GLUE",
                "confidence": 0.9,
            },
        ],
    }


def test_impact_scoring_initialization() -> None:
    tool = HypothesisImpactScoringTool()
    assert tool.name == "hypothesis_impact_scoring"


def test_empty_input() -> None:
    tool = HypothesisImpactScoringTool()
    output = json.loads(tool.execute(""))
    assert output == []

    output2 = json.loads(tool.execute("{}"))
    assert output2 == []


def test_basic_impact_scoring() -> None:
    tool = HypothesisImpactScoringTool()
    data = _make_input()
    output = json.loads(tool.execute(json.dumps(data)))

    assert len(output) == 3
    assert all("impact_score" in s for s in output)
    assert all("hypothesis_type" in s for s in output)
    assert all(0.0 <= s["impact_score"] <= 1.0 for s in output)


def test_contradiction_highest_impact() -> None:
    """Contradictions should score highest due to high novelty."""
    tool = HypothesisImpactScoringTool()
    data = _make_input()
    output = json.loads(tool.execute(json.dumps(data)))

    # Sort by impact score
    sorted_output = sorted(output, key=lambda x: x["impact_score"], reverse=True)

    # Contradiction should be in top positions
    contradiction_scores = [s for s in sorted_output if s["hypothesis_type"] == "contradiction"]
    assert contradiction_scores, "Expected at least one contradiction"
    assert contradiction_scores[0]["impact_score"] > 0.5


def test_gap_high_novelty() -> None:
    """Gaps (missing connections) should score high on novelty."""
    tool = HypothesisImpactScoringTool()
    data = _make_input()
    output = json.loads(tool.execute(json.dumps(data)))

    gap_hyp = [s for s in output if s["hypothesis_id"] == "hyp-1"][0]
    assert gap_hyp["novelty_signal"] >= 0.5


def test_citation_signal_from_entities() -> None:
    """Hypotheses involving frequently-mentioned entities should score higher."""
    tool = HypothesisImpactScoringTool()
    data = _make_input()
    output = json.loads(tool.execute(json.dumps(data)))

    # BERT is mentioned 42 times, GPT 38 times → high-profile entities
    gap_hyp = [s for s in output if s["hypothesis_id"] == "hyp-1"][0]
    assert gap_hyp["citation_signal"] > 0.4


def test_relation_signal_weighted() -> None:
    """Hypotheses involving high-impact relations should score higher."""
    tool = HypothesisImpactScoringTool()
    data = _make_input()
    output = json.loads(tool.execute(json.dumps(data)))

    # BERT appears in OUTPERFORMS (weight 1.0) and ACHIEVES (weight 0.9)
    hub_hyp = [s for s in output if s["hypothesis_id"] == "hyp-3"][0]
    assert hub_hyp["relation_signal"] > 0.4


def test_determinism() -> None:
    """Same input should produce same output."""
    tool = HypothesisImpactScoringTool()
    data = _make_input()
    input_json = json.dumps(data)

    output1 = json.loads(tool.execute(input_json))
    output2 = json.loads(tool.execute(input_json))

    assert output1 == output2


def test_immutability() -> None:
    """Input data should not be modified."""
    tool = HypothesisImpactScoringTool()
    data = _make_input()
    snapshot = json.dumps(data, sort_keys=True)

    tool.execute(json.dumps(data))

    assert json.dumps(data, sort_keys=True) == snapshot


def test_output_sorted_by_impact() -> None:
    """Output should be sorted by impact_score descending."""
    tool = HypothesisImpactScoringTool()
    data = _make_input()
    output = json.loads(tool.execute(json.dumps(data)))

    scores = [s["impact_score"] for s in output]
    assert scores == sorted(scores, reverse=True)


def test_impact_score_bounds() -> None:
    """All scores should be in [0, 1]."""
    tool = HypothesisImpactScoringTool()
    data = _make_input()
    output = json.loads(tool.execute(json.dumps(data)))

    for score in output:
        assert 0.0 <= score["impact_score"] <= 1.0
        assert 0.0 <= score["novelty_signal"] <= 1.0
        assert 0.0 <= score["citation_signal"] <= 1.0
        assert 0.0 <= score["relation_signal"] <= 1.0


def test_provenance_metadata() -> None:
    """Provenance should explain scoring decisions."""
    tool = HypothesisImpactScoringTool()
    data = _make_input()
    output = json.loads(tool.execute(json.dumps(data)))

    for score in output:
        assert "provenance" in score
        prov = score["provenance"]
        assert "hypothesis_type" in prov
        assert "type_weight" in prov
        assert "base_confidence" in prov
        assert "novelty_components" in prov
        assert "citation_components" in prov
        assert "relation_components" in prov


def test_no_relations_graceful() -> None:
    """Should handle missing relations gracefully."""
    tool = HypothesisImpactScoringTool()
    data = _make_input()
    del data["relations"]
    output = json.loads(tool.execute(json.dumps(data)))

    assert len(output) == 3
    assert all("impact_score" in s for s in output)


def test_no_entity_mentions_graceful() -> None:
    """Should handle missing entity mentions gracefully."""
    tool = HypothesisImpactScoringTool()
    data = _make_input()
    del data["entity_mentions"]
    output = json.loads(tool.execute(json.dumps(data)))

    assert len(output) == 3
    assert all("impact_score" in s for s in output)


def test_error_handling() -> None:
    """Invalid input should return error JSON."""
    tool = HypothesisImpactScoringTool()

    # Invalid JSON
    output = json.loads(tool.execute("not json"))
    assert "error" in output

    # Empty hypotheses list should still work
    output2 = json.loads(tool.execute('{"hypotheses": []}'))
    assert output2 == []


def test_rare_entities_boost_novelty() -> None:
    """Hypotheses involving rarely-mentioned entities should boost novelty."""
    tool = HypothesisImpactScoringTool()

    data = {
        "hypotheses": [
            {
                "hypothesis_id": "rare-hyp",
                "hypothesis_type": "gap",
                "description": "Nodes 'RareTech' and 'NewMethod' share connections",
                "confidence": 0.8,
                "supporting_nodes": ["n1", "n2"],
            }
        ],
        "entity_mentions": {
            "RareTech": {"mention_count": 1, "entity_type": "METHOD"},
            "NewMethod": {"mention_count": 1, "entity_type": "METHOD"},
        },
        "relations": [],
    }

    output = json.loads(tool.execute(json.dumps(data)))
    assert len(output) == 1
    # Rare entities should boost novelty
    assert output[0]["novelty_signal"] > 0.6


def test_high_impact_relations_recognized() -> None:
    """OUTPERFORMS and CONTRADICTS relations should have highest weights."""
    tool = HypothesisImpactScoringTool()

    data = {
        "hypotheses": [
            {
                "hypothesis_id": "h1",
                "hypothesis_type": "chain",
                "description": "Chain: 'ModelA' → 'ModelB' → 'ModelC'",
                "confidence": 0.7,
                "supporting_nodes": ["n1", "n2", "n3"],
            }
        ],
        "entity_mentions": {
            "ModelA": {"mention_count": 5, "entity_type": "METHOD"},
            "ModelB": {"mention_count": 5, "entity_type": "METHOD"},
            "ModelC": {"mention_count": 5, "entity_type": "METHOD"},
        },
        "relations": [
            {
                "relation_type": "OUTPERFORMS",
                "subject_span": "ModelA",
                "object_span": "ModelB",
                "confidence": 0.95,
            },
            {
                "relation_type": "CONTRADICTS",
                "subject_span": "ModelB",
                "object_span": "ModelC",
                "confidence": 0.9,
            },
        ],
    }

    output = json.loads(tool.execute(json.dumps(data)))
    assert len(output) == 1
    # Should recognize high-impact relations
    assert output[0]["relation_signal"] > 0.5
