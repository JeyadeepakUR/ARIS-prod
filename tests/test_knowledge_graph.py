"""Test suite for Module 8: Knowledge Graph & Cross-Document Linking."""

import pytest
import uuid as uuid_module
from dataclasses import replace
from datetime import datetime, UTC
from typing import List

from aris.knowledge_graph import (
    Node,
    Edge,
    KnowledgeGraph,
    LinkCandidate,
    Linker,
    LinkMaterializer,
    build_graph_from_corpus,
    add_edges_to_graph,
)
from aris.document_ingestion import Document, DocumentCorpus
from aris.reasoning_engine import ReasoningEngine, ReasoningResult
from aris.evaluation import Evaluator, EvaluationResult
from aris.input_interface import InputPacket, InputInterface


# -----------------------------------------------------------------------------
# Helper Functions for Testing
# -----------------------------------------------------------------------------

def make_doc(content: str, source: str, doc_id_num: int, metadata: dict[str, str] | None = None) -> Document:
    """Helper to create Document instances for testing."""
    return Document(
        content=content,
        source=source,
        format="txt",
        metadata=metadata or {},
        document_id=uuid_module.UUID(f"00000000-0000-0000-0000-{doc_id_num:012d}"),
        created_at=datetime.now(UTC),
    )


# -----------------------------------------------------------------------------
# Mock Implementations for Testing
# -----------------------------------------------------------------------------



class MockReasoningEngine(ReasoningEngine):
    """Mock reasoning engine that returns predefined responses."""
    
    def __init__(self, response_map: dict[str, ReasoningResult] | None = None):
        self.response_map = response_map or {}
        self.call_count = 0
        self.last_packet: InputPacket | None = None
    
    def reason(self, packet: InputPacket) -> ReasoningResult:
        self.call_count += 1
        self.last_packet = packet
        
        # Return mapped response if available
        key = packet.text
        if key in self.response_map:
            return self.response_map[key]
        
        # Default response
        reasoning_steps = [
            f"Step 1: Analyzed link candidate from {packet.text[:50]}",
            "Step 2: Identified relevant connections between documents",
            "Step 3: Link is justified based on shared context",
        ]
        return ReasoningResult(
            reasoning_steps=reasoning_steps,
            confidence_score=0.8,
            input_packet=packet,
        )


class MockEvaluator(Evaluator):
    """Mock evaluator that returns predefined scores."""
    
    def __init__(self, score: float = 0.8):
        self.score = score
        self.call_count = 0
        self.last_result: ReasoningResult | None = None
    
    def evaluate(self, result: ReasoningResult) -> EvaluationResult:
        self.call_count += 1
        self.last_result = result
        
        return EvaluationResult(
            step_count_score=self.score,
            coherence_score=self.score,
            confidence_alignment_score=self.score,
            input_coverage_score=self.score,
            overall_score=self.score,
        )


# -----------------------------------------------------------------------------
# Test Node Dataclass
# -----------------------------------------------------------------------------

def test_node_creation():
    """Test basic node creation."""
    node_id = uuid_module.uuid4()
    doc_id = uuid_module.uuid4()
    node = Node(
        node_id=node_id,
        document_id=doc_id,
        label="Document A",
        node_type="document",
        metadata={"key": "value"},
    )
    assert node.node_id == node_id
    assert node.document_id == doc_id
    assert node.label == "Document A"
    assert node.node_type == "document"
    assert node.metadata == {"key": "value"}


def test_node_immutability():
    """Test that nodes are immutable."""
    node = Node(
        node_id=uuid_module.uuid4(),
        document_id=uuid_module.uuid4(),
        label="Document A",
        node_type="document",
        metadata={},
    )
    with pytest.raises(Exception):
        node.node_id = uuid_module.uuid4()  # type: ignore


def test_node_with_empty_metadata():
    """Test node creation with empty metadata."""
    node = Node(
        node_id=uuid_module.uuid4(),
        document_id=uuid_module.uuid4(),
        label="Document A",
        node_type="document",
        metadata={},
    )
    assert node.metadata == {}


# -----------------------------------------------------------------------------
# Test Edge Dataclass
# -----------------------------------------------------------------------------

def test_edge_creation():
    """Test basic edge creation."""
    edge_id = uuid_module.uuid4()
    source_id = uuid_module.uuid4()
    target_id = uuid_module.uuid4()
    trace_id = uuid_module.uuid4()
    created_at = datetime(2024, 1, 1, 12, 0, 0, tzinfo=UTC)
    edge = Edge(
        edge_id=edge_id,
        source_id=source_id,
        target_id=target_id,
        edge_type="citation",
        evidence="Evidence text",
        reasoning_trace_id=trace_id,
        confidence=0.85,
        metadata={"key": "value"},
        created_at=created_at,
    )
    assert edge.edge_id == edge_id
    assert edge.source_id == source_id
    assert edge.target_id == target_id
    assert edge.edge_type == "citation"
    assert edge.evidence == "Evidence text"
    assert edge.reasoning_trace_id == trace_id
    assert edge.confidence == 0.85
    assert edge.metadata == {"key": "value"}
    assert edge.created_at == created_at


def test_edge_immutability():
    """Test that edges are immutable."""
    edge = Edge(
        edge_id=uuid_module.uuid4(),
        source_id=uuid_module.uuid4(),
        target_id=uuid_module.uuid4(),
        edge_type="citation",
        evidence="Evidence",
        reasoning_trace_id=uuid_module.uuid4(),
        confidence=0.8,
        metadata={},
        created_at=datetime.now(UTC),
    )
    with pytest.raises(Exception):
        edge.confidence = 0.9  # type: ignore


def test_edge_with_evidence():
    """Test that edges must include evidence."""
    edge = Edge(
        edge_id=uuid_module.uuid4(),
        source_id=uuid_module.uuid4(),
        target_id=uuid_module.uuid4(),
        edge_type="citation",
        evidence="This is the evidence",
        reasoning_trace_id=uuid_module.uuid4(),
        confidence=0.8,
        metadata={},
        created_at=datetime.now(UTC),
    )
    assert len(edge.evidence) > 0
    assert "evidence" in edge.evidence.lower()


# -----------------------------------------------------------------------------
# Test KnowledgeGraph Dataclass
# -----------------------------------------------------------------------------

def test_knowledge_graph_creation():
    """Test basic knowledge graph creation."""
    graph_id = uuid_module.uuid4()
    node1_id = uuid_module.uuid4()
    node2_id = uuid_module.uuid4()
    doc1_id = uuid_module.uuid4()
    doc2_id = uuid_module.uuid4()
    nodes = [
        Node(node1_id, doc1_id, "Doc A", "document", {}),
        Node(node2_id, doc2_id, "Doc B", "document", {}),
    ]
    edges = [
        Edge(uuid_module.uuid4(), node1_id, node2_id, "link", "evidence", uuid_module.uuid4(), 0.8, {}, datetime.now(UTC)),
    ]
    created_at = datetime(2024, 1, 1, 12, 0, 0, tzinfo=UTC)
    graph = KnowledgeGraph(
        graph_id=graph_id,
        nodes=nodes,
        edges=edges,
        created_at=created_at,
    )
    assert graph.graph_id == graph_id
    assert len(graph.nodes) == 2
    assert len(graph.edges) == 1
    assert graph.created_at == created_at


def test_knowledge_graph_immutability():
    """Test that knowledge graphs are immutable."""
    graph = KnowledgeGraph(
        graph_id=uuid_module.uuid4(),
        nodes=[],
        edges=[],
        created_at=datetime.now(UTC),
    )
    with pytest.raises(Exception):
        graph.graph_id = uuid_module.uuid4()  # type: ignore


def test_knowledge_graph_empty():
    """Test creation of empty knowledge graph."""
    graph = KnowledgeGraph(
        graph_id=uuid_module.uuid4(),
        nodes=[],
        edges=[],
        created_at=datetime.now(UTC),
    )
    assert len(graph.nodes) == 0
    assert len(graph.edges) == 0


# -----------------------------------------------------------------------------
# Test LinkCandidate Dataclass
# -----------------------------------------------------------------------------

def test_link_candidate_creation():
    """Test basic link candidate creation."""
    doc1_id = uuid_module.uuid4()
    doc2_id = uuid_module.uuid4()
    candidate = LinkCandidate(
        source_document_id=doc1_id,
        target_document_id=doc2_id,
        link_type="citation",
        hint="Both mention 'quantum computing'",
        metadata={"shared_terms": "quantum,computing"},
    )
    assert candidate.source_document_id == doc1_id
    assert candidate.target_document_id == doc2_id
    assert candidate.link_type == "citation"
    assert candidate.hint == "Both mention 'quantum computing'"
    assert "shared_terms" in candidate.metadata


def test_link_candidate_immutability():
    """Test that link candidates are immutable."""
    candidate = LinkCandidate(
        source_document_id=uuid_module.uuid4(),
        target_document_id=uuid_module.uuid4(),
        link_type="citation",
        hint="hint",
        metadata={},
    )
    with pytest.raises(Exception):
        candidate.link_type = "reference"  # type: ignore


# -----------------------------------------------------------------------------
# Test Linker - Keyword Overlap Strategy
# -----------------------------------------------------------------------------

def test_linker_keyword_overlap_basic():
    """Test keyword overlap strategy with shared keywords."""
    corpus = DocumentCorpus(
        documents=[
            make_doc("Document about quantum computing", "source_1", 1),
            make_doc("Study of quantum mechanics", "source_2", 2),
            make_doc("Classical computing systems", "source_3", 3),
        ],
        corpus_id=uuid_module.uuid4(),
        created_at=datetime.now(UTC),
    )
    
    linker = Linker()
    candidates = linker.generate_candidates(corpus, strategy="keyword_overlap", keyword="quantum")
    
    # Should find link between doc_1 and doc_2 (both contain "quantum")
    assert len(candidates) >= 1
    
    doc_1_id = uuid_module.UUID('00000000-0000-0000-0000-000000000001')
    doc_2_id = uuid_module.UUID('00000000-0000-0000-0000-000000000002')
    candidate_pairs = {(c.source_document_id, c.target_document_id) for c in candidates}
    assert (doc_1_id, doc_2_id) in candidate_pairs or (doc_2_id, doc_1_id) in candidate_pairs
    
    for candidate in candidates:
        assert candidate.link_type == "keyword_related"
        assert "quantum" in candidate.hint.lower()


def test_linker_keyword_overlap_no_matches():
    """Test keyword overlap with no matching documents."""
    corpus = DocumentCorpus(
        documents=[
            make_doc("Document about cats", "source_1", 1),
            make_doc("Study of dogs", "source_2", 2),
        ],
        corpus_id=uuid_module.uuid4(),
        created_at=datetime.now(UTC),
    )
    
    linker = Linker()
    candidates = linker.generate_candidates(corpus, strategy="keyword_overlap", keyword="quantum")
    
    assert len(candidates) == 0


def test_linker_keyword_overlap_case_insensitive():
    """Test that keyword matching is case-insensitive."""
    corpus = DocumentCorpus(
        documents=[
            make_doc("QUANTUM computing", "source_1", 1),
            make_doc("quantum mechanics", "source_2", 2),
        ],
        corpus_id=uuid_module.uuid4(),
        created_at=datetime.now(UTC),
    )
    
    linker = Linker()
    candidates = linker.generate_candidates(corpus, strategy="keyword_overlap", keyword="QuAnTuM")
    
    assert len(candidates) >= 1


def test_linker_keyword_overlap_deterministic():
    """Test that keyword overlap strategy is deterministic."""
    corpus = DocumentCorpus(
        documents=[
            make_doc("AI and ML", "source_1", 1),
            make_doc("ML applications", "source_2", 2),
            make_doc("Deep ML", "source_3", 3),
        ],
        corpus_id=uuid_module.uuid4(),
        created_at=datetime.now(UTC),
    )
    
    linker = Linker()
    candidates1 = linker.generate_candidates(corpus, strategy="keyword_overlap", keyword="ML")
    candidates2 = linker.generate_candidates(corpus, strategy="keyword_overlap", keyword="ML")
    
    # Should produce same candidates
    assert len(candidates1) == len(candidates2)
    pairs1 = {(c.source_document_id, c.target_document_id) for c in candidates1}
    pairs2 = {(c.source_document_id, c.target_document_id) for c in candidates2}
    assert pairs1 == pairs2


# -----------------------------------------------------------------------------
# Test Linker - Metadata Match Strategy
# -----------------------------------------------------------------------------

def test_linker_metadata_match_basic():
    """Test metadata match strategy."""
    corpus = DocumentCorpus(
        documents=[
            make_doc("Content A", "source_1", 1, {"author": "Smith"}),
            make_doc("Content B", "source_2", 2, {"author": "Smith"}),
            make_doc("Content C", "source_3", 3, {"author": "Jones"}),
        ],
        corpus_id=uuid_module.uuid4(),
        created_at=datetime.now(UTC),
    )
    
    linker = Linker()
    candidates = linker.generate_candidates(
        corpus,
        strategy="metadata_match",
        metadata_field="author",
        metadata_value="Smith",
    )
    
    assert len(candidates) >= 1
    
    doc_1_id = uuid_module.UUID('00000000-0000-0000-0000-000000000001')
    doc_2_id = uuid_module.UUID('00000000-0000-0000-0000-000000000002')
    candidate_pairs = {(c.source_document_id, c.target_document_id) for c in candidates}
    assert (doc_1_id, doc_2_id) in candidate_pairs or (doc_2_id, doc_1_id) in candidate_pairs
    
    for candidate in candidates:
        assert candidate.link_type == "metadata_related"
        assert "author" in candidate.hint.lower()


def test_linker_metadata_match_no_matches():
    """Test metadata match with no matching documents."""
    corpus = DocumentCorpus(
        documents=[
            make_doc("Content", "source_1", 1, {"author": "Smith"}),
            make_doc("Content", "source_2", 2, {"author": "Jones"}),
        ],
        corpus_id=uuid_module.uuid4(),
        created_at=datetime.now(UTC),
    )
    
    linker = Linker()
    candidates = linker.generate_candidates(
        corpus,
        strategy="metadata_match",
        metadata_field="author",
        metadata_value="Brown",
    )
    
    assert len(candidates) == 0


def test_linker_metadata_match_missing_field():
    """Test metadata match when field doesn't exist."""
    corpus = DocumentCorpus(
        documents=[
            make_doc("Content", "source_1", 1, {"author": "Smith"}),
            make_doc("Content", "source_2", 2, {}),
        ],
        corpus_id=uuid_module.uuid4(),
        created_at=datetime.now(UTC),
    )
    
    linker = Linker()
    candidates = linker.generate_candidates(
        corpus,
        strategy="metadata_match",
        metadata_field="year",
        metadata_value="2024",
    )
    
    assert len(candidates) == 0


# -----------------------------------------------------------------------------
# Test Linker - Sequential Strategy
# -----------------------------------------------------------------------------

def test_linker_sequential_basic():
    """Test sequential strategy creates ordered links."""
    corpus = DocumentCorpus(
        documents=[
            make_doc("First", "source_1", 1),
            make_doc("Second", "source_2", 2),
            make_doc("Third", "source_3", 3),
        ],
        corpus_id=uuid_module.uuid4(),
        created_at=datetime.now(UTC),
    )
    
    linker = Linker()
    candidates = linker.generate_candidates(corpus, strategy="sequential")
    
    assert len(candidates) == 2
    
    # Check order: doc_1 -> doc_2 -> doc_3
    doc_1_id = uuid_module.UUID('00000000-0000-0000-0000-000000000001')
    doc_2_id = uuid_module.UUID('00000000-0000-0000-0000-000000000002')
    doc_3_id = uuid_module.UUID('00000000-0000-0000-0000-000000000003')
    pairs = [(c.source_document_id, c.target_document_id) for c in candidates]
    assert (doc_1_id, doc_2_id) in pairs
    assert (doc_2_id, doc_3_id) in pairs
    
    for candidate in candidates:
        assert candidate.link_type == "sequential"
        assert "sequential" in candidate.hint.lower()


def test_linker_sequential_single_document():
    """Test sequential strategy with single document."""
    corpus = DocumentCorpus(
        documents=[
            make_doc("Only one", "source_1", 1),
        ],
        corpus_id=uuid_module.uuid4(),
        created_at=datetime.now(UTC),
    )
    
    linker = Linker()
    candidates = linker.generate_candidates(corpus, strategy="sequential")
    
    assert len(candidates) == 0


def test_linker_sequential_empty_corpus():
    """Test sequential strategy with empty corpus."""
    corpus = DocumentCorpus(
        documents=[],
        corpus_id=uuid_module.uuid4(),
        created_at=datetime.now(UTC),
    )
    
    linker = Linker()
    candidates = linker.generate_candidates(corpus, strategy="sequential")
    
    assert len(candidates) == 0


def test_linker_sequential_deterministic():
    """Test that sequential strategy is deterministic."""
    corpus = DocumentCorpus(
        documents=[
            make_doc("A", "source_1", 1),
            make_doc("B", "source_2", 2),
            make_doc("C", "source_3", 3),
        ],
        corpus_id=uuid_module.uuid4(),
        created_at=datetime.now(UTC),
    )
    
    linker = Linker()
    candidates1 = linker.generate_candidates(corpus, strategy="sequential")
    candidates2 = linker.generate_candidates(corpus, strategy="sequential")
    
    pairs1 = [(c.source_document_id, c.target_document_id) for c in candidates1]
    pairs2 = [(c.source_document_id, c.target_document_id) for c in candidates2]
    assert pairs1 == pairs2


# -----------------------------------------------------------------------------
# Test Linker - Invalid Strategy
# -----------------------------------------------------------------------------

def test_linker_invalid_strategy():
    """Test that invalid strategy raises ValueError."""
    corpus = DocumentCorpus(
        documents=[
            make_doc("Content", "source_1", 1),
        ],
        corpus_id=uuid_module.uuid4(),
        created_at=datetime.now(UTC),
    )
    
    linker = Linker()
    with pytest.raises(ValueError, match="Unsupported strategy"):
        linker.generate_candidates(corpus, strategy="invalid_strategy")


# -----------------------------------------------------------------------------
# Test LinkMaterializer - Basic Edge Creation
# -----------------------------------------------------------------------------

def test_link_materializer_basic():
    """Test basic link materialization."""
    doc1_id = uuid_module.UUID('00000000-0000-0000-0000-000000000001')
    doc2_id = uuid_module.UUID('00000000-0000-0000-0000-000000000002')
    corpus = DocumentCorpus(
        documents=[
            make_doc("Source document content", "source_1", 1),
            make_doc("Target document content", "source_2", 2),
        ],
        corpus_id=uuid_module.uuid4(),
        created_at=datetime.now(UTC),
    )
    
    candidate = LinkCandidate(
        source_document_id=doc1_id,
        target_document_id=doc2_id,
        link_type="test",
        hint="Test hint",
        metadata={},
    )
    
    reasoning_engine = MockReasoningEngine()
    evaluator = MockEvaluator(score=0.8)
    
    materializer = LinkMaterializer(reasoning_engine, evaluator)
    edge, reasoning_result, eval_result = materializer.materialize(candidate, corpus)
    
    assert edge is not None
    assert edge.source_id == doc1_id
    assert edge.target_id == doc2_id
    assert edge.edge_type == "test"
    assert len(edge.evidence) > 0
    assert edge.confidence == 0.8
    assert reasoning_engine.call_count == 1
    assert evaluator.call_count == 1


def test_link_materializer_evidence_from_reasoning():
    """Test that evidence comes from reasoning steps."""
    doc1_id = uuid_module.UUID('00000000-0000-0000-0000-000000000001')
    doc2_id = uuid_module.UUID('00000000-0000-0000-0000-000000000002')
    corpus = DocumentCorpus(
        documents=[
            make_doc("Source", "source_1", 1),
            make_doc("Target", "source_2", 2),
        ],
        corpus_id=uuid_module.uuid4(),
        created_at=datetime.now(UTC),
    )
    
    candidate = LinkCandidate(
        source_document_id=doc1_id,
        target_document_id=doc2_id,
        link_type="test",
        hint="hint",
        metadata={},
    )
    
    reasoning_engine = MockReasoningEngine()
    evaluator = MockEvaluator(score=0.75)
    
    materializer = LinkMaterializer(reasoning_engine, evaluator)
    edge, reasoning_result, eval_result = materializer.materialize(candidate, corpus)
    
    assert edge is not None
    # Evidence should contain reasoning steps
    for step in reasoning_result.reasoning_steps:
        assert step in edge.evidence


def test_link_materializer_confidence_from_evaluation():
    """Test that confidence comes from evaluation score."""
    doc1_id = uuid_module.UUID('00000000-0000-0000-0000-000000000001')
    doc2_id = uuid_module.UUID('00000000-0000-0000-0000-000000000002')
    corpus = DocumentCorpus(
        documents=[
            make_doc("Source", "source_1", 1),
            make_doc("Target", "source_2", 2),
        ],
        corpus_id=uuid_module.uuid4(),
        created_at=datetime.now(UTC),
    )
    
    candidate = LinkCandidate(
        source_document_id=doc1_id,
        target_document_id=doc2_id,
        link_type="test",
        hint="hint",
        metadata={},
    )
    
    reasoning_engine = MockReasoningEngine()
    evaluator = MockEvaluator(score=0.65)
    
    materializer = LinkMaterializer(reasoning_engine, evaluator)
    edge, reasoning_result, eval_result = materializer.materialize(candidate, corpus)
    
    assert edge is not None
    assert edge.confidence == 0.65


def test_link_materializer_low_confidence_rejected():
    """Test that low confidence scores reject the link."""
    doc1_id = uuid_module.UUID('00000000-0000-0000-0000-000000000001')
    doc2_id = uuid_module.UUID('00000000-0000-0000-0000-000000000002')
    corpus = DocumentCorpus(
        documents=[
            make_doc("Source", "source_1", 1),
            make_doc("Target", "source_2", 2),
        ],
        corpus_id=uuid_module.uuid4(),
        created_at=datetime.now(UTC),
    )
    
    candidate = LinkCandidate(
        source_document_id=doc1_id,
        target_document_id=doc2_id,
        link_type="test",
        hint="hint",
        metadata={},
    )
    
    reasoning_engine = MockReasoningEngine()
    evaluator = MockEvaluator(score=0.3)  # Below threshold
    
    materializer = LinkMaterializer(reasoning_engine, evaluator)
    edge, reasoning_result, eval_result = materializer.materialize(candidate, corpus)
    
    assert edge is None


def test_link_materializer_missing_source_document():
    """Test handling of missing source document."""
    doc1_id = uuid_module.UUID('00000000-0000-0000-0000-000000000001')
    doc2_id = uuid_module.UUID('00000000-0000-0000-0000-000000000002')
    corpus = DocumentCorpus(
        documents=[
            make_doc("Target", "source_2", 2),
        ],
        corpus_id=uuid_module.uuid4(),
        created_at=datetime.now(UTC),
    )
    
    candidate = LinkCandidate(
        source_document_id=doc1_id,  # Doesn't exist
        target_document_id=doc2_id,
        link_type="test",
        hint="hint",
        metadata={},
    )
    
    reasoning_engine = MockReasoningEngine()
    evaluator = MockEvaluator(score=0.8)
    
    materializer = LinkMaterializer(reasoning_engine, evaluator)
    
    with pytest.raises(ValueError, match="not found in corpus"):
        materializer.materialize(candidate, corpus)


def test_link_materializer_missing_target_document():
    """Test handling of missing target document."""
    doc1_id = uuid_module.UUID('00000000-0000-0000-0000-000000000001')
    doc2_id = uuid_module.UUID('00000000-0000-0000-0000-000000000002')
    corpus = DocumentCorpus(
        documents=[
            make_doc("Source", "source_1", 1),
        ],
        corpus_id=uuid_module.uuid4(),
        created_at=datetime.now(UTC),
    )
    
    candidate = LinkCandidate(
        source_document_id=doc1_id,
        target_document_id=doc2_id,  # Doesn't exist
        link_type="test",
        hint="hint",
        metadata={},
    )
    
    reasoning_engine = MockReasoningEngine()
    evaluator = MockEvaluator(score=0.8)
    
    materializer = LinkMaterializer(reasoning_engine, evaluator)
    
    with pytest.raises(ValueError, match="not found in corpus"):
        materializer.materialize(candidate, corpus)


# -----------------------------------------------------------------------------
# Test LinkMaterializer - Batch Operations
# -----------------------------------------------------------------------------

def test_link_materializer_batch_basic():
    """Test batch materialization of multiple candidates."""
    doc1_id = uuid_module.UUID('00000000-0000-0000-0000-000000000001')
    doc2_id = uuid_module.UUID('00000000-0000-0000-0000-000000000002')
    doc3_id = uuid_module.UUID('00000000-0000-0000-0000-000000000003')
    corpus = DocumentCorpus(
        documents=[
            make_doc("First", "source_1", 1),
            make_doc("Second", "source_2", 2),
            make_doc("Third", "source_3", 3),
        ],
        corpus_id=uuid_module.uuid4(),
        created_at=datetime.now(UTC),
    )
    
    candidates = [
        LinkCandidate(doc1_id, doc2_id, "test", "hint1", {}),
        LinkCandidate(doc2_id, doc3_id, "test", "hint2", {}),
    ]
    
    reasoning_engine = MockReasoningEngine()
    evaluator = MockEvaluator(score=0.8)
    
    materializer = LinkMaterializer(reasoning_engine, evaluator)
    edges, failures, rejections = materializer.materialize_batch(candidates, corpus)
    
    assert len(edges) == 2
    assert len(failures) == 0
    assert len(rejections) == 0
    assert reasoning_engine.call_count == 2
    assert evaluator.call_count == 2


def test_link_materializer_batch_with_failures():
    """Test that batch failures are isolated."""
    doc1_id = uuid_module.UUID('00000000-0000-0000-0000-000000000001')
    doc2_id = uuid_module.UUID('00000000-0000-0000-0000-000000000002')
    doc999_id = uuid_module.UUID('00000000-0000-0000-0000-000000000999')
    corpus = DocumentCorpus(
        documents=[
            make_doc("First", "source_1", 1),
            make_doc("Second", "source_2", 2),
        ],
        corpus_id=uuid_module.uuid4(),
        created_at=datetime.now(UTC),
    )
    
    candidates = [
        LinkCandidate(doc1_id, doc2_id, "test", "hint1", {}),
        LinkCandidate(doc1_id, doc999_id, "test", "hint2", {}),  # Missing target
        LinkCandidate(doc2_id, doc1_id, "test", "hint3", {}),
    ]
    
    reasoning_engine = MockReasoningEngine()
    evaluator = MockEvaluator(score=0.8)
    
    materializer = LinkMaterializer(reasoning_engine, evaluator)
    edges, failures, rejections = materializer.materialize_batch(candidates, corpus)
    
    assert len(edges) == 2  # doc_1->doc_2 and doc_2->doc_1
    assert len(failures) == 1  # doc_1->doc_999
    assert len(rejections) == 0


def test_link_materializer_batch_with_rejections():
    """Test that low-confidence links are tracked as rejections."""
    doc1_id = uuid_module.UUID('00000000-0000-0000-0000-000000000001')
    doc2_id = uuid_module.UUID('00000000-0000-0000-0000-000000000002')
    corpus = DocumentCorpus(
        documents=[
            make_doc("First", "source_1", 1),
            make_doc("Second", "source_2", 2),
        ],
        corpus_id=uuid_module.uuid4(),
        created_at=datetime.now(UTC),
    )
    
    candidates = [
        LinkCandidate(doc1_id, doc2_id, "test", "hint1", {}),
    ]
    
    reasoning_engine = MockReasoningEngine()
    evaluator = MockEvaluator(score=0.3)  # Low score
    
    materializer = LinkMaterializer(reasoning_engine, evaluator)
    edges, failures, rejections = materializer.materialize_batch(candidates, corpus)
    
    assert len(edges) == 0
    assert len(failures) == 0
    assert len(rejections) == 1
    assert rejections[0][0] == candidates[0]


def test_link_materializer_batch_empty():
    """Test batch with empty candidate list."""
    corpus = DocumentCorpus(
        documents=[
            make_doc("Content", "source_1", 1),
        ],
        corpus_id=uuid_module.uuid4(),
        created_at=datetime.now(UTC),
    )
    
    reasoning_engine = MockReasoningEngine()
    evaluator = MockEvaluator(score=0.8)
    
    materializer = LinkMaterializer(reasoning_engine, evaluator)
    edges, failures, rejections = materializer.materialize_batch([], corpus)
    
    assert len(edges) == 0
    assert len(failures) == 0
    assert len(rejections) == 0
    assert reasoning_engine.call_count == 0


# -----------------------------------------------------------------------------
# Test Helper Functions
# -----------------------------------------------------------------------------

def test_build_graph_from_corpus():
    """Test building a graph from a corpus (nodes only)."""
    corpus = DocumentCorpus(
        documents=[
            make_doc("Content 1", "source_1", 1, {"key": "val1"}),
            make_doc("Content 2", "source_2", 2, {"key": "val2"}),
        ],
        corpus_id=uuid_module.uuid4(),
        created_at=datetime.now(UTC),
    )
    
    graph = build_graph_from_corpus(corpus)
    
    assert graph.graph_id is not None
    assert len(graph.nodes) == 2
    assert len(graph.edges) == 0
    
    # Check that nodes reference the correct documents
    doc_ids = {d.document_id for d in corpus.documents}
    node_doc_ids = {n.document_id for n in graph.nodes}
    assert node_doc_ids == doc_ids
    
    for node in graph.nodes:
        assert node.node_type == "document"


def test_add_edges_to_graph():
    """Test adding edges to an existing graph (immutable)."""
    graph_id = uuid_module.uuid4()
    node1_id = uuid_module.uuid4()
    node2_id = uuid_module.uuid4()
    doc1_id = uuid_module.uuid4()
    doc2_id = uuid_module.uuid4()
    edge_id = uuid_module.uuid4()
    original_graph = KnowledgeGraph(
        graph_id=graph_id,
        nodes=[
            Node(node1_id, doc1_id, "Doc 1", "document", {}),
            Node(node2_id, doc2_id, "Doc 2", "document", {}),
        ],
        edges=[],
        created_at=datetime.now(UTC),
    )
    
    new_edges = [
        Edge(edge_id, node1_id, node2_id, "link", "evidence", uuid_module.uuid4(), 0.8, {}, datetime.now(UTC)),
    ]
    
    updated_graph = add_edges_to_graph(original_graph, new_edges)
    
    # Original unchanged
    assert len(original_graph.edges) == 0
    
    # New graph has edges
    assert len(updated_graph.edges) == 1
    assert updated_graph.edges[0].edge_id == edge_id
    assert updated_graph.graph_id != original_graph.graph_id  # New graph gets new ID
    assert updated_graph.nodes == original_graph.nodes


def test_add_edges_to_graph_preserves_existing():
    """Test that adding edges preserves existing edges."""
    node1_id = uuid_module.uuid4()
    node2_id = uuid_module.uuid4()
    doc1_id = uuid_module.uuid4()
    doc2_id = uuid_module.uuid4()
    edge0_id = uuid_module.uuid4()
    edge1_id = uuid_module.uuid4()
    existing_edge = Edge(edge0_id, node1_id, node2_id, "old", "evidence", uuid_module.uuid4(), 0.7, {}, datetime.now(UTC))
    
    original_graph = KnowledgeGraph(
        graph_id=uuid_module.uuid4(),
        nodes=[
            Node(node1_id, doc1_id, "Doc 1", "document", {}),
            Node(node2_id, doc2_id, "Doc 2", "document", {}),
        ],
        edges=[existing_edge],
        created_at=datetime.now(UTC),
    )
    
    new_edges = [
        Edge(edge1_id, node2_id, node1_id, "new", "evidence", uuid_module.uuid4(), 0.8, {}, datetime.now(UTC)),
    ]
    
    updated_graph = add_edges_to_graph(original_graph, new_edges)
    
    assert len(updated_graph.edges) == 2
    edge_ids = {e.edge_id for e in updated_graph.edges}
    assert edge0_id in edge_ids
    assert edge1_id in edge_ids


# -----------------------------------------------------------------------------
# Test End-to-End Pipeline
# -----------------------------------------------------------------------------

def test_end_to_end_graph_construction():
    """Test complete pipeline: corpus -> candidates -> edges -> graph."""
    # Create corpus
    corpus = DocumentCorpus(
        documents=[
            make_doc("Research on quantum computing", "source_1", 1),
            make_doc("Quantum mechanics applications", "source_2", 2),
            make_doc("Classical algorithms", "source_3", 3),
        ],
        corpus_id=uuid_module.uuid4(),
        created_at=datetime.now(UTC),
    )
    
    # Generate candidates
    linker = Linker()
    candidates = linker.generate_candidates(corpus, strategy="keyword_overlap", keyword="quantum")
    
    assert len(candidates) >= 1
    
    # Materialize edges
    reasoning_engine = MockReasoningEngine()
    evaluator = MockEvaluator(score=0.8)
    
    materializer = LinkMaterializer(reasoning_engine, evaluator)
    edges, failures, rejections = materializer.materialize_batch(candidates, corpus)
    
    assert len(edges) >= 1
    assert len(failures) == 0
    
    # Build graph
    graph = build_graph_from_corpus(corpus)
    final_graph = add_edges_to_graph(graph, edges)
    
    assert len(final_graph.nodes) == 3
    assert len(final_graph.edges) >= 1
    
    # Verify edge properties
    for edge in final_graph.edges:
        assert len(edge.evidence) > 0
        assert edge.confidence >= 0.5
        assert edge.reasoning_trace_id is not None


def test_end_to_end_sequential_linking():
    """Test sequential linking pipeline."""
    corpus = DocumentCorpus(
        documents=[
            make_doc("Part 1", "source_1", 1),
            make_doc("Part 2", "source_2", 2),
            make_doc("Part 3", "source_3", 3),
        ],
        corpus_id=uuid_module.uuid4(),
        created_at=datetime.now(UTC),
    )
    
    linker = Linker()
    candidates = linker.generate_candidates(corpus, strategy="sequential")
    
    assert len(candidates) == 2
    
    reasoning_engine = MockReasoningEngine()
    evaluator = MockEvaluator(score=0.9)
    
    materializer = LinkMaterializer(reasoning_engine, evaluator)
    edges, failures, rejections = materializer.materialize_batch(candidates, corpus)
    
    assert len(edges) == 2
    assert len(failures) == 0
    assert len(rejections) == 0
    
    # Check ordering
    doc_1_id = uuid_module.UUID('00000000-0000-0000-0000-000000000001')
    doc_2_id = uuid_module.UUID('00000000-0000-0000-0000-000000000002')
    doc_3_id = uuid_module.UUID('00000000-0000-0000-0000-000000000003')
    edge_pairs = [(e.source_id, e.target_id) for e in edges]
    assert (doc_1_id, doc_2_id) in edge_pairs
    assert (doc_2_id, doc_3_id) in edge_pairs


def test_end_to_end_metadata_linking():
    """Test metadata-based linking pipeline."""
    corpus = DocumentCorpus(
        documents=[
            make_doc("Paper A", "source_1", 1, {"author": "Smith", "year": "2024"}),
            make_doc("Paper B", "source_2", 2, {"author": "Smith", "year": "2023"}),
            make_doc("Paper C", "source_3", 3, {"author": "Jones", "year": "2024"}),
        ],
        corpus_id=uuid_module.uuid4(),
        created_at=datetime.now(UTC),
    )
    
    linker = Linker()
    candidates = linker.generate_candidates(
        corpus,
        strategy="metadata_match",
        metadata_field="author",
        metadata_value="Smith",
    )
    
    assert len(candidates) >= 1
    
    reasoning_engine = MockReasoningEngine()
    evaluator = MockEvaluator(score=0.85)
    
    materializer = LinkMaterializer(reasoning_engine, evaluator)
    edges, failures, rejections = materializer.materialize_batch(candidates, corpus)
    
    assert len(edges) >= 1
    
    graph = build_graph_from_corpus(corpus)
    final_graph = add_edges_to_graph(graph, edges)
    
    # Should only link Smith's papers
    doc_1_id = uuid_module.UUID('00000000-0000-0000-0000-000000000001')
    doc_2_id = uuid_module.UUID('00000000-0000-0000-0000-000000000002')
    for edge in final_graph.edges:
        assert edge.source_id in [doc_1_id, doc_2_id]
        assert edge.target_id in [doc_1_id, doc_2_id]


# -----------------------------------------------------------------------------
# Test Determinism
# -----------------------------------------------------------------------------

def test_full_pipeline_determinism():
    """Test that the entire pipeline is deterministic."""
    corpus = DocumentCorpus(
        documents=[
            make_doc("AI and ML", "source_1", 1),
            make_doc("ML models", "source_2", 2),
            make_doc("Deep ML", "source_3", 3),
        ],
        corpus_id=uuid_module.uuid4(),
        created_at=datetime(2024, 1, 1, 12, 0, 0, tzinfo=UTC),
    )
    
    linker = Linker()
    reasoning_engine = MockReasoningEngine()
    evaluator = MockEvaluator(score=0.8)
    materializer = LinkMaterializer(reasoning_engine, evaluator)
    
    # Run 1
    candidates1 = linker.generate_candidates(corpus, strategy="keyword_overlap", keyword="ML")
    edges1, _, _ = materializer.materialize_batch(candidates1, corpus)
    graph1 = build_graph_from_corpus(corpus)
    final1 = add_edges_to_graph(graph1, edges1)
    
    # Run 2
    candidates2 = linker.generate_candidates(corpus, strategy="keyword_overlap", keyword="ML")
    edges2, _, _ = materializer.materialize_batch(candidates2, corpus)
    graph2 = build_graph_from_corpus(corpus)
    final2 = add_edges_to_graph(graph2, edges2)
    
    # Compare - document IDs should be same since we use deterministic UUIDs
    assert len(final1.nodes) == len(final2.nodes)
    assert len(final1.edges) == len(final2.edges)
    
    # Note: node_ids are generated with uuid4() so they won't match between runs
    # But document_ids will match since we use make_doc() with deterministic IDs
    doc_ids1 = {n.document_id for n in final1.nodes}
    doc_ids2 = {n.document_id for n in final2.nodes}
    assert doc_ids1 == doc_ids2
    
    edge_pairs1 = {(e.source_id, e.target_id) for e in final1.edges}
    edge_pairs2 = {(e.source_id, e.target_id) for e in final2.edges}
    assert edge_pairs1 == edge_pairs2
