"""Tests for cross-domain bridge discovery."""

from aris.core.semantic_analyzer import SemanticAnalyzer
from aris.graph.bridge_discovery import BridgeDiscoveryEngine


def test_bridge_concepts_found_between_domains() -> None:
    analyzer = SemanticAnalyzer()
    engine = BridgeDiscoveryEngine()

    profile_nlp = analyzer.analyze(
        "Attention improves retrieval and transformer architecture enables better ranking.",
        document_id="nlp-1",
        domain="nlp",
    )
    profile_bio = analyzer.analyze(
        "Attention mechanism improves protein structure ranking in biology pipelines.",
        document_id="bio-1",
        domain="biology",
    )

    bridges = engine.discover([profile_nlp, profile_bio], top_k=10)

    assert len(bridges) >= 1
    assert bridges[0].novelty >= 0.08
    domains = {bridges[0].source_domain, bridges[0].target_domain}
    assert domains == {"biology", "nlp"}
