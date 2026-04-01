"""Tests for deterministic semantic analysis."""

from aris.core.semantic_analyzer import SemanticAnalyzer


def test_semantic_profile_contains_keywords_and_claims() -> None:
    analyzer = SemanticAnalyzer()
    text = (
        "Transformer model improves retrieval quality by 12%. "
        "However baseline cannot generalize to out-of-domain settings."
    )

    profile = analyzer.analyze(text, document_id="doc-1", domain="nlp")

    assert profile.document_id == "doc-1"
    assert profile.domain == "nlp"
    assert len(profile.keywords) > 0
    assert len(profile.claims) >= 2
    polarities = {claim.polarity for claim in profile.claims}
    assert "positive" in polarities
    assert "negative" in polarities


def test_semantic_profile_is_deterministic() -> None:
    analyzer = SemanticAnalyzer()
    text = "Graph method outperforms baseline and supports robust transfer learning."

    p1 = analyzer.analyze(text, document_id="doc-x", domain="ml")
    p2 = analyzer.analyze(text, document_id="doc-x", domain="ml")

    assert p1 == p2
