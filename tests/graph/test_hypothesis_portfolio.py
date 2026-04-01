"""Tests for deep/wide hypothesis portfolio generation."""

from aris.core.semantic_analyzer import SemanticAnalyzer
from aris.graph.bridge_discovery import BridgeDiscoveryEngine
from aris.graph.contradiction_engine import ContradictionEngine
from aris.graph.hypothesis_portfolio import HypothesisPortfolioEngine


def test_portfolio_contains_deep_and_wide_modes() -> None:
    analyzer = SemanticAnalyzer()

    docs = [
        analyzer.analyze(
            "Method X improves diagnosis accuracy and supports low-noise settings.",
            document_id="paper-1",
            domain="medical_ai",
        ),
        analyzer.analyze(
            "Method X fails diagnosis in noisy settings and contradicts prior stability claims.",
            document_id="paper-2",
            domain="medical_ai",
        ),
        analyzer.analyze(
            "Method X attention strategy improves ranking in legal document search.",
            document_id="paper-3",
            domain="legal_nlp",
        ),
    ]

    contradiction_engine = ContradictionEngine()
    bridge_engine = BridgeDiscoveryEngine()
    portfolio_engine = HypothesisPortfolioEngine()

    contradictions = contradiction_engine.find_contradictions(docs)
    bridges = bridge_engine.discover(docs, top_k=10)
    portfolio = portfolio_engine.generate(docs, contradictions, bridges, max_items=20)

    assert len(portfolio) >= 1
    modes = {item.mode for item in portfolio}
    assert "deep" in modes
    if bridges:
        assert "wide" in modes
    assert all(item.overall_priority >= 0.0 for item in portfolio)
    assert all(item.overall_priority <= 1.0 for item in portfolio)
