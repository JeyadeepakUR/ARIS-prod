"""Tests for cross-paper contradiction detection."""

from aris.core.semantic_analyzer import SemanticAnalyzer
from aris.graph.contradiction_engine import ContradictionEngine


def test_contradictions_detected_for_opposing_claims() -> None:
    analyzer = SemanticAnalyzer()
    engine = ContradictionEngine()

    profile_a = analyzer.analyze(
        "Method Alpha improves survival prediction and supports stable outcomes.",
        document_id="paper-a",
        domain="biomed",
    )
    profile_b = analyzer.analyze(
        "Method Alpha fails survival prediction and contradicts stable outcomes.",
        document_id="paper-b",
        domain="biomed",
    )

    records = engine.find_contradictions([profile_a, profile_b])

    assert len(records) >= 1
    top = records[0]
    assert top.document_a == "paper-a"
    assert top.document_b == "paper-b"
    assert top.severity > 0.0
    assert len(top.overlap_terms) >= 2
