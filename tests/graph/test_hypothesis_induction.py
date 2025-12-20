"""Tests for Descriptive Hypothesis Induction (Module 13)."""

import uuid
from datetime import UTC, datetime

import pytest

from aris.graph.hypothesis_induction import HypothesisCandidate, HypothesisInductionEngine
from aris.graph.knowledge_graph import Edge, KnowledgeGraph, Node


@pytest.fixture
def empty_graph() -> KnowledgeGraph:
    """Empty knowledge graph."""
    return KnowledgeGraph(
        graph_id=uuid.uuid4(),
        nodes=[],
        edges=[],
        created_at=datetime.now(UTC),
    )


@pytest.fixture
def simple_graph() -> KnowledgeGraph:
    """Simple graph with 4 nodes and 3 edges."""
    doc_id = uuid.uuid4()
    node_a = Node(uuid.uuid4(), doc_id, "A", "document", {})
    node_b = Node(uuid.uuid4(), doc_id, "B", "document", {})
    node_c = Node(uuid.uuid4(), doc_id, "C", "document", {})
    node_d = Node(uuid.uuid4(), doc_id, "D", "document", {})

    edge_ab = Edge(
        uuid.uuid4(),
        node_a.node_id,
        node_b.node_id,
        "references",
        "A references B",
        uuid.uuid4(),
        0.9,
        {},
        datetime.now(UTC),
    )
    edge_bc = Edge(
        uuid.uuid4(),
        node_b.node_id,
        node_c.node_id,
        "references",
        "B references C",
        uuid.uuid4(),
        0.8,
        {},
        datetime.now(UTC),
    )
    edge_cd = Edge(
        uuid.uuid4(),
        node_c.node_id,
        node_d.node_id,
        "references",
        "C references D",
        uuid.uuid4(),
        0.85,
        {},
        datetime.now(UTC),
    )

    return KnowledgeGraph(
        graph_id=uuid.uuid4(),
        nodes=[node_a, node_b, node_c, node_d],
        edges=[edge_ab, edge_bc, edge_cd],
        created_at=datetime.now(UTC),
    )


@pytest.fixture
def triangle_graph() -> KnowledgeGraph:
    """Graph with a triangle (cluster) pattern."""
    doc_id = uuid.uuid4()
    node_a = Node(uuid.uuid4(), doc_id, "A", "document", {})
    node_b = Node(uuid.uuid4(), doc_id, "B", "document", {})
    node_c = Node(uuid.uuid4(), doc_id, "C", "document", {})

    edge_ab = Edge(
        uuid.uuid4(),
        node_a.node_id,
        node_b.node_id,
        "references",
        "A to B",
        uuid.uuid4(),
        0.9,
        {},
        datetime.now(UTC),
    )
    edge_bc = Edge(
        uuid.uuid4(),
        node_b.node_id,
        node_c.node_id,
        "references",
        "B to C",
        uuid.uuid4(),
        0.9,
        {},
        datetime.now(UTC),
    )
    edge_ca = Edge(
        uuid.uuid4(),
        node_c.node_id,
        node_a.node_id,
        "references",
        "C to A",
        uuid.uuid4(),
        0.9,
        {},
        datetime.now(UTC),
    )

    return KnowledgeGraph(
        graph_id=uuid.uuid4(),
        nodes=[node_a, node_b, node_c],
        edges=[edge_ab, edge_bc, edge_ca],
        created_at=datetime.now(UTC),
    )


@pytest.fixture
def hub_graph() -> KnowledgeGraph:
    """Graph with a hub node."""
    doc_id = uuid.uuid4()
    hub = Node(uuid.uuid4(), doc_id, "Hub", "document", {})
    spoke_a = Node(uuid.uuid4(), doc_id, "SpokeA", "document", {})
    spoke_b = Node(uuid.uuid4(), doc_id, "SpokeB", "document", {})
    spoke_c = Node(uuid.uuid4(), doc_id, "SpokeC", "document", {})

    edges = [
        Edge(
            uuid.uuid4(),
            hub.node_id,
            spoke_a.node_id,
            "references",
            "Hub to A",
            uuid.uuid4(),
            0.9,
            {},
            datetime.now(UTC),
        ),
        Edge(
            uuid.uuid4(),
            hub.node_id,
            spoke_b.node_id,
            "references",
            "Hub to B",
            uuid.uuid4(),
            0.9,
            {},
            datetime.now(UTC),
        ),
        Edge(
            uuid.uuid4(),
            hub.node_id,
            spoke_c.node_id,
            "references",
            "Hub to C",
            uuid.uuid4(),
            0.9,
            {},
            datetime.now(UTC),
        ),
    ]

    return KnowledgeGraph(
        graph_id=uuid.uuid4(),
        nodes=[hub, spoke_a, spoke_b, spoke_c],
        edges=edges,
        created_at=datetime.now(UTC),
    )


@pytest.fixture
def contradiction_graph() -> KnowledgeGraph:
    """Graph with contradictory edges."""
    doc_id = uuid.uuid4()
    node_a = Node(uuid.uuid4(), doc_id, "A", "document", {})
    node_b = Node(uuid.uuid4(), doc_id, "B", "document", {})

    edge_supports = Edge(
        uuid.uuid4(),
        node_a.node_id,
        node_b.node_id,
        "supports",
        "A supports B",
        uuid.uuid4(),
        0.9,
        {},
        datetime.now(UTC),
    )
    edge_contradicts = Edge(
        uuid.uuid4(),
        node_a.node_id,
        node_b.node_id,
        "contradicts",
        "A contradicts B",
        uuid.uuid4(),
        0.8,
        {},
        datetime.now(UTC),
    )

    return KnowledgeGraph(
        graph_id=uuid.uuid4(),
        nodes=[node_a, node_b],
        edges=[edge_supports, edge_contradicts],
        created_at=datetime.now(UTC),
    )


class TestHypothesisCandidate:
    """Test HypothesisCandidate dataclass."""

    def test_immutability(self):
        """Hypothesis candidates are immutable."""
        candidate = HypothesisCandidate(
            hypothesis_id=uuid.uuid4(),
            hypothesis_type="gap",
            description="Test hypothesis",
            supporting_nodes=(uuid.uuid4(),),
            supporting_edges=(uuid.uuid4(),),
            confidence=0.8,
            metadata={"key": "value"},
            created_at=datetime.now(UTC),
        )

        with pytest.raises(Exception):  # FrozenInstanceError or AttributeError
            candidate.confidence = 0.9  # type: ignore

    def test_to_dict(self):
        """to_dict returns JSON-serializable representation."""
        h_id = uuid.uuid4()
        n_id = uuid.uuid4()
        e_id = uuid.uuid4()
        created = datetime.now(UTC)

        candidate = HypothesisCandidate(
            hypothesis_id=h_id,
            hypothesis_type="cluster",
            description="Test cluster",
            supporting_nodes=(n_id,),
            supporting_edges=(e_id,),
            confidence=0.75,
            metadata={"size": 3},
            created_at=created,
        )

        data = candidate.to_dict()

        assert data["hypothesis_id"] == str(h_id)
        assert data["hypothesis_type"] == "cluster"
        assert data["description"] == "Test cluster"
        assert data["supporting_nodes"] == [str(n_id)]
        assert data["supporting_edges"] == [str(e_id)]
        assert data["confidence"] == 0.75
        assert data["metadata"] == {"size": 3}
        assert data["created_at"] == created.isoformat()

    def test_confidence_clamping(self):
        """Confidence is clamped to [0, 1] in to_dict."""
        candidate = HypothesisCandidate(
            hypothesis_id=uuid.uuid4(),
            hypothesis_type="hub",
            description="Over-confident",
            supporting_nodes=(),
            supporting_edges=(),
            confidence=1.5,  # Invalid but allowed in dataclass
            metadata={},
            created_at=datetime.now(UTC),
        )

        data = candidate.to_dict()
        assert data["confidence"] == 1.0

        candidate2 = HypothesisCandidate(
            hypothesis_id=uuid.uuid4(),
            hypothesis_type="hub",
            description="Under-confident",
            supporting_nodes=(),
            supporting_edges=(),
            confidence=-0.1,
            metadata={},
            created_at=datetime.now(UTC),
        )

        data2 = candidate2.to_dict()
        assert data2["confidence"] == 0.0


class TestHypothesisInductionEngine:
    """Test HypothesisInductionEngine."""

    def test_initialization(self):
        """Engine initializes with valid min_confidence."""
        engine = HypothesisInductionEngine(min_confidence=0.6)
        assert engine._min_confidence == 0.6

    def test_invalid_min_confidence(self):
        """Engine rejects invalid min_confidence."""
        with pytest.raises(ValueError, match="min_confidence must be in"):
            HypothesisInductionEngine(min_confidence=1.5)

        with pytest.raises(ValueError, match="min_confidence must be in"):
            HypothesisInductionEngine(min_confidence=-0.1)

    def test_empty_graph(self, empty_graph):
        """No hypotheses from empty graph."""
        engine = HypothesisInductionEngine()
        hypotheses = engine.induce_hypotheses(empty_graph)
        assert len(hypotheses) == 0

    def test_graph_immutability(self, simple_graph):
        """Graph is not modified during hypothesis induction."""
        engine = HypothesisInductionEngine()

        # Snapshot initial state
        node_count = len(simple_graph.nodes)
        edge_count = len(simple_graph.edges)
        node_ids = {n.node_id for n in simple_graph.nodes}
        edge_ids = {e.edge_id for e in simple_graph.edges}

        # Induce hypotheses
        hypotheses = engine.induce_hypotheses(simple_graph)

        # Verify immutability
        assert len(simple_graph.nodes) == node_count
        assert len(simple_graph.edges) == edge_count
        assert {n.node_id for n in simple_graph.nodes} == node_ids
        assert {e.edge_id for e in simple_graph.edges} == edge_ids

    def test_determinism(self, simple_graph):
        """Same graph produces same hypotheses."""
        engine = HypothesisInductionEngine(min_confidence=0.0)

        hypotheses1 = engine.induce_hypotheses(simple_graph)
        hypotheses2 = engine.induce_hypotheses(simple_graph)

        # Same count
        assert len(hypotheses1) == len(hypotheses2)

        # Sort by type and description for comparison
        h1_sorted = sorted(hypotheses1, key=lambda h: (h.hypothesis_type, h.description))
        h2_sorted = sorted(hypotheses2, key=lambda h: (h.hypothesis_type, h.description))

        for h1, h2 in zip(h1_sorted, h2_sorted):
            assert h1.hypothesis_type == h2.hypothesis_type
            assert h1.description == h2.description
            assert h1.confidence == h2.confidence
            assert h1.supporting_nodes == h2.supporting_nodes
            assert h1.supporting_edges == h2.supporting_edges

    def test_detect_chain(self, simple_graph):
        """Detect chain pattern (A→B→C→D)."""
        engine = HypothesisInductionEngine(min_confidence=0.0)
        hypotheses = engine.induce_hypotheses(simple_graph)

        chain_hypotheses = [h for h in hypotheses if h.hypothesis_type == "chain"]
        assert len(chain_hypotheses) >= 1

        # Check one chain hypothesis
        chain = chain_hypotheses[0]
        assert chain.hypothesis_type == "chain"
        assert len(chain.supporting_nodes) >= 3
        assert len(chain.supporting_edges) >= 2
        assert 0.0 <= chain.confidence <= 1.0

    def test_detect_cluster(self, triangle_graph):
        """Detect cluster pattern (triangle)."""
        engine = HypothesisInductionEngine(min_confidence=0.0)
        hypotheses = engine.induce_hypotheses(triangle_graph)

        cluster_hypotheses = [h for h in hypotheses if h.hypothesis_type == "cluster"]
        assert len(cluster_hypotheses) >= 1

        cluster = cluster_hypotheses[0]
        assert cluster.hypothesis_type == "cluster"
        assert len(cluster.supporting_nodes) == 3
        assert len(cluster.supporting_edges) == 3
        assert cluster.confidence == 1.0  # Perfect triangle

    def test_detect_hub(self, hub_graph):
        """Detect hub pattern."""
        engine = HypothesisInductionEngine(min_confidence=0.0)
        hypotheses = engine.induce_hypotheses(hub_graph)

        hub_hypotheses = [h for h in hypotheses if h.hypothesis_type == "hub"]
        assert len(hub_hypotheses) >= 1

        hub = hub_hypotheses[0]
        assert hub.hypothesis_type == "hub"
        assert "hub" in hub.description.lower()
        assert len(hub.supporting_edges) >= 3

    def test_detect_contradiction(self, contradiction_graph):
        """Detect contradiction pattern."""
        engine = HypothesisInductionEngine(min_confidence=0.0)
        hypotheses = engine.induce_hypotheses(contradiction_graph)

        contradiction_hypotheses = [h for h in hypotheses if h.hypothesis_type == "contradiction"]
        assert len(contradiction_hypotheses) >= 1

        contradiction = contradiction_hypotheses[0]
        assert contradiction.hypothesis_type == "contradiction"
        assert "contradict" in contradiction.description.lower()
        assert len(contradiction.supporting_edges) == 2
        assert contradiction.confidence == 1.0  # Definitive

    def test_detect_gap(self):
        """Detect gap pattern (shared neighbors)."""
        doc_id = uuid.uuid4()
        node_a = Node(uuid.uuid4(), doc_id, "A", "document", {})
        node_b = Node(uuid.uuid4(), doc_id, "B", "document", {})
        node_c = Node(uuid.uuid4(), doc_id, "C", "document", {})
        node_d = Node(uuid.uuid4(), doc_id, "D", "document", {})

        # A and B both connect to C and D, but not to each other
        edges = [
            Edge(
                uuid.uuid4(),
                node_a.node_id,
                node_c.node_id,
                "references",
                "",
                uuid.uuid4(),
                0.9,
                {},
                datetime.now(UTC),
            ),
            Edge(
                uuid.uuid4(),
                node_a.node_id,
                node_d.node_id,
                "references",
                "",
                uuid.uuid4(),
                0.9,
                {},
                datetime.now(UTC),
            ),
            Edge(
                uuid.uuid4(),
                node_b.node_id,
                node_c.node_id,
                "references",
                "",
                uuid.uuid4(),
                0.9,
                {},
                datetime.now(UTC),
            ),
            Edge(
                uuid.uuid4(),
                node_b.node_id,
                node_d.node_id,
                "references",
                "",
                uuid.uuid4(),
                0.9,
                {},
                datetime.now(UTC),
            ),
        ]

        graph = KnowledgeGraph(
            graph_id=uuid.uuid4(),
            nodes=[node_a, node_b, node_c, node_d],
            edges=edges,
            created_at=datetime.now(UTC),
        )

        engine = HypothesisInductionEngine(min_confidence=0.0)
        hypotheses = engine.induce_hypotheses(graph)

        gap_hypotheses = [h for h in hypotheses if h.hypothesis_type == "gap"]
        assert len(gap_hypotheses) >= 1

        gap = gap_hypotheses[0]
        assert gap.hypothesis_type == "gap"
        assert "share" in gap.description.lower()
        assert len(gap.supporting_nodes) >= 4  # A, B, C, D

    def test_confidence_bounds(self, simple_graph):
        """All hypotheses have confidence in [0, 1]."""
        engine = HypothesisInductionEngine(min_confidence=0.0)
        hypotheses = engine.induce_hypotheses(simple_graph)

        for h in hypotheses:
            assert 0.0 <= h.confidence <= 1.0

    def test_min_confidence_filter(self, simple_graph):
        """min_confidence filters low-confidence hypotheses."""
        engine_low = HypothesisInductionEngine(min_confidence=0.0)
        engine_high = HypothesisInductionEngine(min_confidence=0.9)

        hypotheses_low = engine_low.induce_hypotheses(simple_graph)
        hypotheses_high = engine_high.induce_hypotheses(simple_graph)

        assert len(hypotheses_high) <= len(hypotheses_low)

        for h in hypotheses_high:
            assert h.confidence >= 0.9

    def test_closed_hypothesis_types(self, simple_graph):
        """All hypotheses use closed set of types."""
        engine = HypothesisInductionEngine(min_confidence=0.0)
        hypotheses = engine.induce_hypotheses(simple_graph)

        for h in hypotheses:
            assert h.hypothesis_type in HypothesisInductionEngine.HYPOTHESIS_TYPES

    def test_no_prediction(self, simple_graph):
        """Hypotheses are descriptive, not predictive."""
        engine = HypothesisInductionEngine()
        hypotheses = engine.induce_hypotheses(simple_graph)

        # Check that descriptions are observation-based
        for h in hypotheses:
            # No future tense or predictions
            forbidden_words = ["will", "should", "predict", "expect", "forecast"]
            assert not any(word in h.description.lower() for word in forbidden_words)

    def test_hypothesis_correctness_chain(self, simple_graph):
        """Chain hypothesis correctly identifies sequential connections."""
        engine = HypothesisInductionEngine(min_confidence=0.0)
        hypotheses = engine.induce_hypotheses(simple_graph)

        chain_hypotheses = [h for h in hypotheses if h.hypothesis_type == "chain"]
        assert len(chain_hypotheses) > 0

        # Verify chain structure
        chain = chain_hypotheses[0]
        nodes = list(chain.supporting_nodes)
        edges = list(chain.supporting_edges)

        # Verify edges connect sequential nodes
        edge_dict = {e.edge_id: e for e in simple_graph.edges}
        for i in range(len(edges)):
            edge = edge_dict[edges[i]]
            # Edge should connect consecutive nodes in the chain
            assert edge.source_id in nodes
            assert edge.target_id in nodes

    def test_hypothesis_correctness_cluster(self, triangle_graph):
        """Cluster hypothesis correctly identifies dense subgraph."""
        engine = HypothesisInductionEngine(min_confidence=0.0)
        hypotheses = engine.induce_hypotheses(triangle_graph)

        cluster_hypotheses = [h for h in hypotheses if h.hypothesis_type == "cluster"]
        assert len(cluster_hypotheses) > 0

        cluster = cluster_hypotheses[0]
        nodes = set(cluster.supporting_nodes)
        edges = set(cluster.supporting_edges)

        # Verify all edges are internal to the cluster
        edge_dict = {e.edge_id: e for e in triangle_graph.edges}
        for edge_id in edges:
            edge = edge_dict[edge_id]
            assert edge.source_id in nodes
            assert edge.target_id in nodes
