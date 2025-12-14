"""Test suite for Module 8: Knowledge Graph & Cross-Document Linking."""

import pytest
import uuid as uuid_module
from datetime import datetime, UTC

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


# Helper to create documents
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


# Mock implementations
class MockReasoningEngine(ReasoningEngine):
    """Mock reasoning engine."""
    def __init__(self, score: float = 0.8):
        self.score = score
        self.call_count = 0
    
    def reason(self, packet: InputPacket) -> ReasoningResult:
        self.call_count += 1
        return ReasoningResult(
            reasoning_steps=["Step 1", "Step 2", "Step 3"],
            confidence_score=self.score,
            input_packet=packet,
        )


class MockEvaluator(Evaluator):
    """Mock evaluator."""
    def __init__(self, score: float = 0.8):
        self.score = score
        self.call_count = 0
    
    def evaluate(self, result: ReasoningResult) -> EvaluationResult:
        self.call_count += 1
        return EvaluationResult(
            step_count_score=self.score,
            coherence_score=self.score,
            confidence_alignment_score=self.score,
            input_coverage_score=self.score,
            overall_score=self.score,
        )


# Dataclass tests
def test_node_creation():
    """Test Node creation and immutability."""
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
    assert node.label == "Document A"
    assert node.node_type == "document"
    
    # Test immutability
    with pytest.raises(Exception):
        node.label = "Document B"  # type: ignore


def test_edge_creation():
    """Test Edge creation with evidence."""
    edge_id = uuid_module.uuid4()
    source_id = uuid_module.uuid4()
    target_id = uuid_module.uuid4()
    edge = Edge(
        edge_id=edge_id,
        source_id=source_id,
        target_id=target_id,
        edge_type="citation",
        evidence="This is evidence",
        reasoning_trace_id=uuid_module.uuid4(),
        confidence=0.85,
        metadata={},
        created_at=datetime.now(UTC),
    )
    assert edge.source_id == source_id
    assert edge.target_id == target_id
    assert edge.confidence == 0.85
    assert len(edge.evidence) > 0


def test_knowledge_graph_creation():
    """Test KnowledgeGraph creation."""
    nodes = [
        Node(uuid_module.uuid4(), uuid_module.uuid4(), "Doc A", "document", {}),
        Node(uuid_module.uuid4(), uuid_module.uuid4(), "Doc B", "document", {}),
    ]
    graph = KnowledgeGraph(
        graph_id=uuid_module.uuid4(),
        nodes=nodes,
        edges=[],
        created_at=datetime.now(UTC),
    )
    assert len(graph.nodes) == 2
    assert len(graph.edges) == 0


def test_link_candidate_creation():
    """Test LinkCandidate creation."""
    doc1_id = uuid_module.uuid4()
    doc2_id = uuid_module.uuid4()
    candidate = LinkCandidate(
        source_document_id=doc1_id,
        target_document_id=doc2_id,
        link_type="citation",
        hint="Both mention quantum",
        metadata={},
    )
    assert candidate.source_document_id == doc1_id
    assert candidate.target_document_id == doc2_id


# Linker tests
def test_linker_keyword_overlap():
    """Test keyword overlap strategy."""
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
    
    assert len(candidates) >= 1
    for candidate in candidates:
        assert candidate.link_type == "keyword_related"


def test_linker_sequential():
    """Test sequential linking strategy."""
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
    for candidate in candidates:
        assert candidate.link_type == "sequential"


def test_linker_metadata_match():
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
    for candidate in candidates:
        assert candidate.link_type == "metadata_related"


def test_linker_invalid_strategy():
    """Test invalid strategy raises ValueError."""
    corpus = DocumentCorpus(
        documents=[make_doc("Content", "source_1", 1)],
        corpus_id=uuid_module.uuid4(),
        created_at=datetime.now(UTC),
    )
    
    linker = Linker()
    with pytest.raises(ValueError, match="Unsupported strategy"):
        linker.generate_candidates(corpus, strategy="invalid_strategy")


def test_materializer_basic():
    """Test basic link materialization."""
    corpus = DocumentCorpus(
        documents=[
            make_doc("Source content", "source_1", 1),
            make_doc("Target content", "source_2", 2),
        ],
        corpus_id=uuid_module.uuid4(),
        created_at=datetime.now(UTC),
    )
    
    candidate = LinkCandidate(
        source_document_id=uuid_module.UUID('00000000-0000-0000-0000-000000000001'),
        target_document_id=uuid_module.UUID('00000000-0000-0000-0000-000000000002'),
        link_type="test",
        hint="hint",
        metadata={},
    )
    
    reasoning_engine = MockReasoningEngine(score=0.9)
    evaluator = MockEvaluator(score=0.9)
    
    materializer = LinkMaterializer(reasoning_engine, evaluator)
    edge, reasoning_result, eval_result = materializer.materialize(candidate, corpus)
    
    assert edge is not None
    assert edge.confidence >= 0.5
    assert len(edge.evidence) > 0


def test_materializer_low_confidence():
    """Test that low confidence rejects the edge."""
    corpus = DocumentCorpus(
        documents=[
            make_doc("Source", "source_1", 1),
            make_doc("Target", "source_2", 2),
        ],
        corpus_id=uuid_module.uuid4(),
        created_at=datetime.now(UTC),
    )
    
    candidate = LinkCandidate(
        source_document_id=uuid_module.UUID('00000000-0000-0000-0000-000000000001'),
        target_document_id=uuid_module.UUID('00000000-0000-0000-0000-000000000002'),
        link_type="test",
        hint="hint",
        metadata={},
    )
    
    reasoning_engine = MockReasoningEngine()
    evaluator = MockEvaluator(score=0.3)  # Below threshold
    
    materializer = LinkMaterializer(reasoning_engine, evaluator)
    edge, _, _ = materializer.materialize(candidate, corpus)
    
    assert edge is None


def test_materializer_batch():
    """Test batch materialization."""
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
        LinkCandidate(
            uuid_module.UUID('00000000-0000-0000-0000-000000000001'),
            uuid_module.UUID('00000000-0000-0000-0000-000000000002'),
            "test",
            "hint1",
            {},
        ),
        LinkCandidate(
            uuid_module.UUID('00000000-0000-0000-0000-000000000002'),
            uuid_module.UUID('00000000-0000-0000-0000-000000000003'),
            "test",
            "hint2",
            {},
        ),
    ]
    
    reasoning_engine = MockReasoningEngine()
    evaluator = MockEvaluator(score=0.8)
    
    materializer = LinkMaterializer(reasoning_engine, evaluator)
    edges, failures, rejections = materializer.materialize_batch(candidates, corpus)
    
    assert len(edges) == 2
    assert len(failures) == 0


# Helper function tests
def test_build_graph_from_corpus():
    """Test graph building from corpus."""
    corpus = DocumentCorpus(
        documents=[
            make_doc("Content 1", "source_1", 1),
            make_doc("Content 2", "source_2", 2),
        ],
        corpus_id=uuid_module.uuid4(),
        created_at=datetime.now(UTC),
    )
    
    graph = build_graph_from_corpus(corpus)
    
    assert len(graph.nodes) == 2
    assert len(graph.edges) == 0
    for node in graph.nodes:
        assert node.node_type == "document"


def test_add_edges_to_graph():
    """Test immutable edge addition."""
    node1_id = uuid_module.uuid4()
    node2_id = uuid_module.uuid4()
    original_graph = KnowledgeGraph(
        graph_id=uuid_module.uuid4(),
        nodes=[
            Node(node1_id, uuid_module.uuid4(), "Doc 1", "document", {}),
            Node(node2_id, uuid_module.uuid4(), "Doc 2", "document", {}),
        ],
        edges=[],
        created_at=datetime.now(UTC),
    )
    
    new_edges = [
        Edge(uuid_module.uuid4(), node1_id, node2_id, "link", "evidence", uuid_module.uuid4(), 0.8, {}, datetime.now(UTC)),
    ]
    
    updated_graph = add_edges_to_graph(original_graph, new_edges)
    
    # Original unchanged
    assert len(original_graph.edges) == 0
    
    # New graph has edge
    assert len(updated_graph.edges) == 1


# Integration test
def test_end_to_end_pipeline():
    """Test complete pipeline."""
    corpus = DocumentCorpus(
        documents=[
            make_doc("Research on quantum", "source_1", 1),
            make_doc("Quantum mechanics", "source_2", 2),
            make_doc("Classical systems", "source_3", 3),
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
    
    # Build graph
    graph = build_graph_from_corpus(corpus)
    final_graph = add_edges_to_graph(graph, edges)
    
    assert len(final_graph.nodes) == 3
    assert len(final_graph.edges) >= 1
    
    # Verify edges have evidence
    for edge in final_graph.edges:
        assert len(edge.evidence) > 0
        assert edge.confidence >= 0.5
