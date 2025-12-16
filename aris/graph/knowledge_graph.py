"""
Knowledge graph and cross-document linking for ARIS.

Provides deterministic candidate generation and evidence-backed link materialization
using ReasoningEngine for justification and Evaluator for scoring.
"""

import uuid
from dataclasses import dataclass
from datetime import UTC, datetime

from aris.graph.document_ingestion import Document, DocumentCorpus
from aris.core.evaluation import EvaluationResult, Evaluator
from aris.core.input_interface import InputInterface
from aris.core.reasoning_engine import ReasoningEngine, ReasoningResult


@dataclass(frozen=True)
class Node:
    """
    Immutable node in the knowledge graph.

    Attributes:
        node_id: Unique identifier for this node
        document_id: UUID of the source document
        label: Human-readable label (document title, entity name, etc.)
        node_type: Type classification ("document", "entity", "concept", etc.)
        metadata: Additional metadata about the node
    """

    node_id: uuid.UUID
    document_id: uuid.UUID
    label: str
    node_type: str
    metadata: dict[str, str]


@dataclass(frozen=True)
class Edge:
    """
    Immutable directed edge in the knowledge graph.

    Attributes:
        edge_id: Unique identifier for this edge
        source_id: Node ID of the source node
        target_id: Node ID of the target node
        edge_type: Relationship type ("cites", "references", "supports", etc.)
        evidence: Text evidence supporting this link
        reasoning_trace_id: UUID of the reasoning trace that justified this edge
        confidence: Confidence score from evaluation (0-1)
        metadata: Additional metadata about the edge
        created_at: UTC timestamp when edge was created
    """

    edge_id: uuid.UUID
    source_id: uuid.UUID
    target_id: uuid.UUID
    edge_type: str
    evidence: str
    reasoning_trace_id: uuid.UUID
    confidence: float
    metadata: dict[str, str]
    created_at: datetime


@dataclass(frozen=True)
class KnowledgeGraph:
    """
    Immutable knowledge graph with nodes and edges.

    Attributes:
        graph_id: Unique identifier for this graph
        nodes: List of Node objects
        edges: List of Edge objects
        created_at: UTC timestamp when graph was created
    """

    graph_id: uuid.UUID
    nodes: list[Node]
    edges: list[Edge]
    created_at: datetime


@dataclass(frozen=True)
class LinkCandidate:
    """
    Potential link before validation.

    Attributes:
        source_document_id: UUID of source document
        target_document_id: UUID of target document
        link_type: Proposed relationship type
        hint: Optional hint for reasoning (e.g., shared entity, keyword)
        metadata: Additional metadata about the candidate
    """

    source_document_id: uuid.UUID
    target_document_id: uuid.UUID
    link_type: str
    hint: str
    metadata: dict[str, str]


class Linker:
    """
    Generates link candidates using deterministic heuristics.

    No embeddings, no similarity computations—only explicit criteria like
    shared keywords, document structure, or metadata patterns.
    """

    def generate_candidates(
        self,
        corpus: DocumentCorpus,
        strategy: str = "",
        keyword: str = "",
        metadata_field: str = "",
        metadata_value: str = "",
    ) -> list[LinkCandidate]:
        """
        Generate link candidates based on explicit criteria.

        Args:
            corpus: DocumentCorpus to analyze
            strategy: Link generation strategy
                - "keyword_overlap": Find docs sharing a keyword
                - "metadata_match": Find docs with matching metadata field
                - "sequential": Link consecutive documents
            keyword: For keyword_overlap strategy
            metadata_field: For metadata_match strategy
            metadata_value: For metadata_match strategy

        Returns:
            List of LinkCandidate objects

        Raises:
            ValueError: If strategy is unsupported
        """
        criteria = {
            "strategy": strategy,
            "keyword": keyword,
            "field": metadata_field,
            "value": metadata_value,
        }

        if strategy == "keyword_overlap":
            return self._keyword_overlap_candidates(corpus, criteria)
        elif strategy == "metadata_match":
            return self._metadata_match_candidates(corpus, criteria)
        elif strategy == "sequential":
            return self._sequential_candidates(corpus)
        else:
            raise ValueError(
                f"Unsupported strategy: {strategy}. "
                f"Supported: keyword_overlap, metadata_match, sequential"
            )

    def _keyword_overlap_candidates(
        self,
        corpus: DocumentCorpus,
        criteria: dict[str, str],
    ) -> list[LinkCandidate]:
        """Generate candidates based on keyword overlap."""
        keyword = criteria.get("keyword", "").lower()
        if not keyword:
            raise ValueError("keyword_overlap strategy requires 'keyword' parameter")

        candidates: list[LinkCandidate] = []
        docs_with_keyword = [
            doc for doc in corpus.documents if keyword in doc.content.lower()
        ]

        # Create pairwise candidates for all documents containing the keyword
        for i, source_doc in enumerate(docs_with_keyword):
            for target_doc in docs_with_keyword[i + 1 :]:
                candidates.append(
                    LinkCandidate(
                        source_document_id=source_doc.document_id,
                        target_document_id=target_doc.document_id,
                        link_type="keyword_related",
                        hint=f"shared_keyword:{keyword}",
                        metadata={"strategy": "keyword_overlap", "keyword": keyword},
                    )
                )

        return candidates

    def _metadata_match_candidates(
        self,
        corpus: DocumentCorpus,
        criteria: dict[str, str],
    ) -> list[LinkCandidate]:
        """Generate candidates based on metadata field matching."""
        field = criteria.get("field", "")
        if not field:
            raise ValueError("metadata_match strategy requires 'field' parameter")

        candidates: list[LinkCandidate] = []
        # Group documents by metadata field value
        field_groups: dict[str, list[Document]] = {}

        for doc in corpus.documents:
            value = doc.metadata.get(field)
            if value:
                if value not in field_groups:
                    field_groups[value] = []
                field_groups[value].append(doc)

        # Create pairwise candidates within each group
        for value, docs in field_groups.items():
            for i, source_doc in enumerate(docs):
                for target_doc in docs[i + 1 :]:
                    candidates.append(
                        LinkCandidate(
                            source_document_id=source_doc.document_id,
                            target_document_id=target_doc.document_id,
                            link_type="metadata_related",
                            hint=f"shared_{field}:{value}",
                            metadata={
                                "strategy": "metadata_match",
                                "field": field,
                                "value": value,
                            },
                        )
                    )

        return candidates

    def _sequential_candidates(self, corpus: DocumentCorpus) -> list[LinkCandidate]:
        """Generate candidates linking sequential documents."""
        candidates: list[LinkCandidate] = []

        for i in range(len(corpus.documents) - 1):
            source_doc = corpus.documents[i]
            target_doc = corpus.documents[i + 1]

            candidates.append(
                LinkCandidate(
                    source_document_id=source_doc.document_id,
                    target_document_id=target_doc.document_id,
                    link_type="sequential",
                    hint=f"sequential_order:{i}",
                    metadata={"strategy": "sequential", "index": str(i)},
                )
            )

        return candidates


class LinkMaterializer:
    """
    Validates link candidates using ReasoningEngine and Evaluator.

    Each candidate is justified through reasoning, evaluated for quality,
    and materialized into an Edge with evidence and confidence score.
    """

    def __init__(
        self,
        reasoning_engine: ReasoningEngine,
        evaluator: Evaluator,
    ) -> None:
        """
        Initialize materializer with reasoning and evaluation components.

        Args:
            reasoning_engine: ReasoningEngine for link justification
            evaluator: Evaluator for scoring justifications
        """
        self._reasoning_engine = reasoning_engine
        self._evaluator = evaluator
        self._input_interface = InputInterface()

    def materialize(
        self,
        candidate: LinkCandidate,
        corpus: DocumentCorpus,
    ) -> tuple[Edge | None, ReasoningResult, EvaluationResult]:
        """
        Materialize a link candidate into an edge with evidence.

        Process:
        1. Retrieve source and target documents
        2. Create reasoning prompt from candidate + documents
        3. Generate justification via ReasoningEngine
        4. Evaluate justification quality
        5. Create Edge if reasoning is adequate

        Args:
            candidate: LinkCandidate to validate
            corpus: DocumentCorpus containing the documents

        Returns:
            Tuple of (Edge or None, ReasoningResult, EvaluationResult)
            Edge is None if justification is inadequate

        Raises:
            ValueError: If documents not found in corpus
        """
        # Find documents
        source_doc = self._find_document(corpus, candidate.source_document_id)
        target_doc = self._find_document(corpus, candidate.target_document_id)

        if source_doc is None or target_doc is None:
            raise ValueError("Source or target document not found in corpus")

        # Create reasoning prompt
        prompt = self._create_prompt(candidate, source_doc, target_doc)

        # Validate prompt through InputInterface
        packet = self._input_interface.accept(
            prompt,
            source=f"link_candidate:{candidate.link_type}",
        )

        # Generate justification
        reasoning_result = self._reasoning_engine.reason(packet)

        # Evaluate justification
        evaluation_result = self._evaluator.evaluate(reasoning_result)

        # Materialize edge if evaluation is adequate (>= 0.5 overall score)
        edge: Edge | None = None
        if evaluation_result.overall_score >= 0.5:
            # Extract evidence from reasoning steps
            evidence = " | ".join(reasoning_result.reasoning_steps)

            edge = Edge(
                edge_id=uuid.uuid4(),
                source_id=candidate.source_document_id,
                target_id=candidate.target_document_id,
                edge_type=candidate.link_type,
                evidence=evidence,
                reasoning_trace_id=packet.request_id,
                confidence=evaluation_result.overall_score,
                metadata=candidate.metadata,
                created_at=datetime.now(UTC),
            )

        return edge, reasoning_result, evaluation_result

    def materialize_batch(
        self,
        candidates: list[LinkCandidate],
        corpus: DocumentCorpus,
    ) -> tuple[
        list[Edge],
        list[tuple[LinkCandidate, Exception]],
        list[tuple[LinkCandidate, ReasoningResult, EvaluationResult]],
    ]:
        """
        Materialize multiple candidates with failure isolation.

        Args:
            candidates: List of LinkCandidate objects
            corpus: DocumentCorpus containing the documents

        Returns:
            Tuple of:
            - Materialized edges
            - Failed candidates with exceptions
            - Rejected candidates (low confidence) with reasoning/evaluation
        """
        edges: list[Edge] = []
        failures: list[tuple[LinkCandidate, Exception]] = []
        rejections: list[tuple[LinkCandidate, ReasoningResult, EvaluationResult]] = []

        for candidate in candidates:
            try:
                edge, reasoning, evaluation = self.materialize(candidate, corpus)
                if edge is not None:
                    edges.append(edge)
                else:
                    # Low confidence rejection
                    rejections.append((candidate, reasoning, evaluation))
            except Exception as e:
                failures.append((candidate, e))

        return edges, failures, rejections

    def _find_document(
        self, corpus: DocumentCorpus, document_id: uuid.UUID
    ) -> Document | None:
        """Find document by ID in corpus."""
        for doc in corpus.documents:
            if doc.document_id == document_id:
                return doc
        return None

    def _create_prompt(
        self,
        candidate: LinkCandidate,
        source_doc: Document,
        target_doc: Document,
    ) -> str:
        """Create reasoning prompt from candidate and documents."""
        prompt = f"""Analyze the relationship between two documents.

Source Document: {source_doc.source}
Source Content: {source_doc.content[:500]}...

Target Document: {target_doc.source}
Target Content: {target_doc.content[:500]}...

Proposed Link Type: {candidate.link_type}
Hint: {candidate.hint}

Justify whether this link is valid and provide evidence."""

        return prompt


def build_graph_from_corpus(
    corpus: DocumentCorpus,
) -> KnowledgeGraph:
    """
    Build a basic knowledge graph from corpus (nodes only, no edges).

    Creates one node per document. Use LinkMaterializer to add edges.

    Args:
        corpus: DocumentCorpus to convert to graph

    Returns:
        KnowledgeGraph with document nodes
    """
    nodes: list[Node] = []

    for doc in corpus.documents:
        node = Node(
            node_id=uuid.uuid4(),
            document_id=doc.document_id,
            label=doc.source,
            node_type="document",
            metadata=doc.metadata,
        )
        nodes.append(node)

    return KnowledgeGraph(
        graph_id=uuid.uuid4(),
        nodes=nodes,
        edges=[],
        created_at=datetime.now(UTC),
    )


def add_edges_to_graph(
    graph: KnowledgeGraph,
    edges: list[Edge],
) -> KnowledgeGraph:
    """
    Create a new graph with additional edges.

    Args:
        graph: Existing KnowledgeGraph
        edges: List of Edge objects to add

    Returns:
        New KnowledgeGraph with combined edges
    """
    return KnowledgeGraph(
        graph_id=uuid.uuid4(),
        nodes=graph.nodes,
        edges=graph.edges + edges,
        created_at=datetime.now(UTC),
    )
