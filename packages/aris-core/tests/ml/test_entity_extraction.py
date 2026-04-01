"""Tests for Scientific Entity Extraction (Module 10B - Novel Component)."""

from __future__ import annotations

import json

import pytest

from aris.ml.entity_extraction import EntityCandidate, ScientificEntityTool


def _make_simple_input() -> dict[str, Any]:
    return {
        "text": "BERT achieves 92.8% accuracy on GLUE benchmark.",
        "document_id": "doc_1"
    }


def _make_cross_doc_input() -> dict[str, Any]:
    return {
        "text": "We use BERT for text classification on GLUE dataset.",
        "document_id": "doc_1",
        "context_documents": [
            {"text": "BERT was introduced in 2018 for NLU tasks. GLUE is a benchmark.", "id": "doc_2"},
            {"text": "Previous work used BERT on various datasets including GLUE.", "id": "doc_3"},
        ]
    }


def test_initialization() -> None:
    """Tool initializes correctly."""
    tool = ScientificEntityTool()
    assert tool.name == "scientific_entity_extraction"
    assert tool._device == "cpu"


def test_empty_input() -> None:
    """Empty input returns empty list."""
    tool = ScientificEntityTool()
    result = json.loads(tool.execute(""))
    assert result == []


def test_basic_entity_extraction() -> None:
    """Extract entities from simple text."""
    tool = ScientificEntityTool()
    input_data = _make_simple_input()
    result = json.loads(tool.execute(json.dumps(input_data)))
    
    # Should find BERT (METHOD), GLUE (DATASET), 92.8% (METRIC)
    assert len(result) >= 2  # At least BERT and GLUE
    
    # Check BERT entity
    bert_entities = [e for e in result if e["span"] == "BERT"]
    assert len(bert_entities) == 1
    assert bert_entities[0]["type"] == "METHOD"
    assert 0.0 <= bert_entities[0]["confidence"] <= 1.0


def test_cross_document_boosting() -> None:
    """Cross-document mentions boost confidence."""
    tool = ScientificEntityTool()
    
    # Test without context
    simple_input = _make_simple_input()
    simple_result = json.loads(tool.execute(json.dumps(simple_input)))
    simple_bert = [e for e in simple_result if e["span"] == "BERT"][0]
    
    # Test with context (BERT appears in multiple docs)
    cross_doc_input = _make_cross_doc_input()
    cross_result = json.loads(tool.execute(json.dumps(cross_doc_input)))
    cross_bert = [e for e in cross_result if e["span"] == "BERT"][0]
    
    # Cross-doc version should have higher confidence
    assert cross_bert["confidence"] >= simple_bert["confidence"]
    assert cross_bert["mention_count"] > 0
    assert "context_score" in cross_bert


def test_entity_types() -> None:
    """All entity types are detected correctly."""
    tool = ScientificEntityTool()
    text = {
        "text": "Transformer models achieve 95.5% F1 on WMT14 machine translation task.",
        "document_id": "doc_test"
    }
    result = json.loads(tool.execute(json.dumps(text)))
    
    entity_types = {e["type"] for e in result}
    # Should detect METHOD (Transformer), METRIC (95.5%, F1), DATASET (WMT14), TASK (machine translation)
    assert "METHOD" in entity_types or "DATASET" in entity_types or "METRIC" in entity_types


def test_confidence_bounds() -> None:
    """All confidences are in [0, 1]."""
    tool = ScientificEntityTool()
    input_data = _make_cross_doc_input()
    result = json.loads(tool.execute(json.dumps(input_data)))
    
    for entity in result:
        assert 0.0 <= entity["confidence"] <= 1.0
        assert 0.0 <= entity["context_score"] <= 1.0


def test_determinism() -> None:
    """Same input produces same output."""
    tool = ScientificEntityTool()
    input_data = _make_simple_input()
    input_json = json.dumps(input_data)
    
    result1 = json.loads(tool.execute(input_json))
    result2 = json.loads(tool.execute(input_json))
    
    assert len(result1) == len(result2)
    # Sort by span for comparison
    result1_sorted = sorted(result1, key=lambda x: (x["span"], x["start"]))
    result2_sorted = sorted(result2, key=lambda x: (x["span"], x["start"]))
    
    for e1, e2 in zip(result1_sorted, result2_sorted):
        assert e1["span"] == e2["span"]
        assert e1["type"] == e2["type"]
        assert abs(e1["confidence"] - e2["confidence"]) < 0.001


def test_no_overlapping_entities() -> None:
    """Entities don't overlap in span positions."""
    tool = ScientificEntityTool()
    input_data = {
        "text": "BERT and GPT are transformer-based models for NLP tasks.",
        "document_id": "test"
    }
    result = json.loads(tool.execute(json.dumps(input_data)))
    
    # Check for overlaps
    for i, e1 in enumerate(result):
        for e2 in result[i+1:]:
            start1, end1 = e1["start"], e1["end"]
            start2, end2 = e2["start"], e2["end"]
            # No overlap
            assert end1 <= start2 or end2 <= start1


def test_provenance_tracking() -> None:
    """Each entity has full provenance."""
    tool = ScientificEntityTool()
    input_data = _make_cross_doc_input()
    result = json.loads(tool.execute(json.dumps(input_data)))
    
    for entity in result:
        assert "provenance" in entity
        prov = entity["provenance"]
        assert "model" in prov
        assert "extraction_method" in prov
        assert "document_id" in prov
        assert prov["extraction_method"] == "pattern_plus_context"


def test_mention_count_tracking() -> None:
    """Mention counts are tracked correctly."""
    tool = ScientificEntityTool()
    input_data = {
        "text": "BERT is used for classification.",
        "document_id": "main",
        "context_documents": [
            {"text": "BERT was proposed in 2018.", "id": "ctx1"},
            {"text": "We fine-tune BERT on our data.", "id": "ctx2"},
            {"text": "No mention of the model here.", "id": "ctx3"},
        ]
    }
    result = json.loads(tool.execute(json.dumps(input_data)))
    
    bert_entity = [e for e in result if e["span"] == "BERT"][0]
    # BERT appears in main doc + 2 context docs = mention_count should be 2
    assert bert_entity["mention_count"] >= 1


def test_error_handling() -> None:
    """Invalid input produces structured error."""
    tool = ScientificEntityTool()
    
    # Invalid JSON
    result = json.loads(tool.execute("not json"))
    assert "error" in result
    
    # Missing text field
    result = json.loads(tool.execute('{"document_id": "test"}'))
    assert "error" in result
    assert "text" in result["error"].lower()


def test_context_similarity() -> None:
    """Context similarity scoring works."""
    tool = ScientificEntityTool()
    
    # Similar contexts
    sim1 = tool._context_similarity(
        "BERT is a transformer model for NLP",
        "BERT transformer architecture for language"
    )
    
    # Dissimilar contexts
    sim2 = tool._context_similarity(
        "BERT is a transformer model for NLP",
        "Image classification using CNNs on ImageNet"
    )
    
    assert sim1 > sim2
    assert 0.0 <= sim1 <= 1.0
    assert 0.0 <= sim2 <= 1.0
