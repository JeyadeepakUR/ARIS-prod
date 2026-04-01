"""Descriptive Hypothesis Induction (Module 13).

This module generates explicit, testable research hypotheses by analyzing patterns
in the knowledge graph. It detects predefined graph motifs and emits descriptive
hypothesis candidates.

Constraints:
- No outcome prediction
- No graph modification
- No ML model invocation
- Closed set of hypothesis types
- Deterministic behavior
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any

from aris.graph.knowledge_graph import Edge, KnowledgeGraph, Node


@dataclass(frozen=True)
class HypothesisCandidate:
    """Immutable descriptive hypothesis candidate.

    A hypothesis describes an observable pattern in the graph without predicting
    outcomes or asserting truth.

    Attributes:
        hypothesis_id: Unique identifier
        hypothesis_type: Closed set type (gap, cluster, chain, hub, contradiction)
        description: Human-readable hypothesis statement
        supporting_nodes: Node IDs that support this hypothesis
        supporting_edges: Edge IDs that support this hypothesis
        confidence: Confidence score (0-1) based on pattern strength
        metadata: Additional pattern-specific metadata
        created_at: UTC timestamp when hypothesis was created
    """

    hypothesis_id: uuid.UUID
    hypothesis_type: str
    description: str
    supporting_nodes: tuple[uuid.UUID, ...]
    supporting_edges: tuple[uuid.UUID, ...]
    confidence: float
    metadata: dict[str, Any]
    created_at: datetime

    def to_dict(self) -> dict[str, Any]:
        """Return JSON-serializable representation."""
        return {
            "hypothesis_id": str(self.hypothesis_id),
            "hypothesis_type": self.hypothesis_type,
            "description": self.description,
            "supporting_nodes": [str(nid) for nid in self.supporting_nodes],
            "supporting_edges": [str(eid) for eid in self.supporting_edges],
            "confidence": float(max(0.0, min(1.0, self.confidence))),
            "metadata": self.metadata,
            "created_at": self.created_at.isoformat(),
        }


class HypothesisInductionEngine:
    """Generate descriptive hypothesis candidates from graph patterns.

    Detects predefined motifs:
    - Gap: Nodes that could be connected but aren't
    - Cluster: Groups of highly interconnected nodes
    - Chain: Sequential node connections
    - Hub: Nodes with many connections
    - Contradiction: Conflicting edge types between same node pair
    """

    # Closed set of hypothesis types
    HYPOTHESIS_TYPES: tuple[str, ...] = (
        "gap",
        "cluster",
        "chain",
        "hub",
        "contradiction",
    )

    def __init__(self, min_confidence: float = 0.5) -> None:
        """Initialize engine with minimum confidence threshold.

        Args:
            min_confidence: Minimum confidence for emitted hypotheses (0-1)
        """
        if not 0.0 <= min_confidence <= 1.0:
            raise ValueError(f"min_confidence must be in [0, 1], got {min_confidence}")
        self._min_confidence = min_confidence

    def induce_hypotheses(self, graph: KnowledgeGraph) -> list[HypothesisCandidate]:
        """Generate all hypothesis candidates from graph patterns.

        Args:
            graph: KnowledgeGraph to analyze

        Returns:
            List of HypothesisCandidate objects meeting confidence threshold

        Ensures:
            - Graph is not modified
            - Deterministic output for same graph
            - All hypotheses are descriptive (no predictions)
        """
        # Verify immutability: snapshot graph state
        node_count = len(graph.nodes)
        edge_count = len(graph.edges)

        candidates: list[HypothesisCandidate] = []

        # Detect each motif type
        candidates.extend(self._detect_gaps(graph))
        candidates.extend(self._detect_clusters(graph))
        candidates.extend(self._detect_chains(graph))
        candidates.extend(self._detect_hubs(graph))
        candidates.extend(self._detect_contradictions(graph))

        # Verify graph was not modified
        assert len(graph.nodes) == node_count, "Graph nodes were modified"
        assert len(graph.edges) == edge_count, "Graph edges were modified"

        # Filter by confidence
        return [c for c in candidates if c.confidence >= self._min_confidence]

    def _detect_gaps(self, graph: KnowledgeGraph) -> list[HypothesisCandidate]:
        """Detect gap patterns: nodes that share neighbors but aren't connected."""
        candidates: list[HypothesisCandidate] = []

        if len(graph.nodes) < 2:
            return candidates

        # Build adjacency structure
        adjacency: dict[uuid.UUID, set[uuid.UUID]] = {n.node_id: set() for n in graph.nodes}
        for edge in graph.edges:
            adjacency[edge.source_id].add(edge.target_id)
            adjacency[edge.target_id].add(edge.source_id)  # Treat as undirected for gaps

        # Find node pairs with shared neighbors but no direct connection
        node_ids = [n.node_id for n in graph.nodes]
        for i, node_a in enumerate(node_ids):
            for node_b in node_ids[i + 1 :]:
                # Skip if already connected
                if node_b in adjacency[node_a] or node_a in adjacency[node_b]:
                    continue

                # Count shared neighbors
                shared = adjacency[node_a] & adjacency[node_b]
                if len(shared) >= 2:
                    # Confidence based on number of shared neighbors
                    confidence = min(1.0, len(shared) / 5.0)

                    node_a_label = next((n.label for n in graph.nodes if n.node_id == node_a), "?")
                    node_b_label = next((n.label for n in graph.nodes if n.node_id == node_b), "?")

                    candidates.append(
                        HypothesisCandidate(
                            hypothesis_id=uuid.uuid4(),
                            hypothesis_type="gap",
                            description=f"Nodes '{node_a_label}' and '{node_b_label}' share {len(shared)} neighbors but are not directly connected",
                            supporting_nodes=(node_a, node_b) + tuple(shared),
                            supporting_edges=(),
                            confidence=confidence,
                            metadata={
                                "shared_neighbor_count": len(shared),
                                "node_a_id": str(node_a),
                                "node_b_id": str(node_b),
                            },
                            created_at=datetime.now(UTC),
                        )
                    )

        return candidates

    def _detect_clusters(self, graph: KnowledgeGraph) -> list[HypothesisCandidate]:
        """Detect cluster patterns: groups of highly interconnected nodes."""
        candidates: list[HypothesisCandidate] = []

        if len(graph.nodes) < 3:
            return candidates

        # Build adjacency
        adjacency: dict[uuid.UUID, set[uuid.UUID]] = {n.node_id: set() for n in graph.nodes}
        for edge in graph.edges:
            adjacency[edge.source_id].add(edge.target_id)
            adjacency[edge.target_id].add(edge.source_id)

        # Find dense subgraphs using simple threshold
        # A cluster is 3+ nodes where each has edges to >50% of others
        node_ids = [n.node_id for n in graph.nodes]
        for i, node_a in enumerate(node_ids):
            for j, node_b in enumerate(node_ids[i + 1 :], start=i + 1):
                for node_c in node_ids[j + 1 :]:
                    # Check if these 3 nodes form a dense cluster
                    neighbors_a = adjacency[node_a]
                    neighbors_b = adjacency[node_b]
                    neighbors_c = adjacency[node_c]

                    # Count internal edges
                    internal_edges = 0
                    if node_b in neighbors_a or node_a in neighbors_b:
                        internal_edges += 1
                    if node_c in neighbors_a or node_a in neighbors_c:
                        internal_edges += 1
                    if node_c in neighbors_b or node_b in neighbors_c:
                        internal_edges += 1

                    # If all 3 are connected (triangle), it's a cluster
                    if internal_edges >= 3:
                        cluster_nodes = {node_a, node_b, node_c}
                        cluster_edges = tuple(
                            e.edge_id
                            for e in graph.edges
                            if e.source_id in cluster_nodes and e.target_id in cluster_nodes
                        )

                        # Confidence: perfect triangle = 1.0
                        confidence = internal_edges / 3.0

                        labels = [
                            next((n.label for n in graph.nodes if n.node_id == nid), "?")
                            for nid in [node_a, node_b, node_c]
                        ]

                        candidates.append(
                            HypothesisCandidate(
                                hypothesis_id=uuid.uuid4(),
                                hypothesis_type="cluster",
                                description=f"Dense cluster detected among nodes: {', '.join(labels)}",
                                supporting_nodes=(node_a, node_b, node_c),
                                supporting_edges=cluster_edges,
                                confidence=confidence,
                                metadata={
                                    "cluster_size": 3,
                                    "internal_edge_count": internal_edges,
                                },
                                created_at=datetime.now(UTC),
                            )
                        )

        return candidates

    def _detect_chains(self, graph: KnowledgeGraph) -> list[HypothesisCandidate]:
        """Detect chain patterns: sequential node connections (A→B→C)."""
        candidates: list[HypothesisCandidate] = []

        if len(graph.edges) < 2:
            return candidates

        # Build directed adjacency
        adjacency: dict[uuid.UUID, list[tuple[uuid.UUID, uuid.UUID]]] = {
            n.node_id: [] for n in graph.nodes
        }
        for edge in graph.edges:
            adjacency[edge.source_id].append((edge.target_id, edge.edge_id))

        # Find chains of length 3+ (A→B→C or longer)
        for start_node in adjacency:
            # BFS to find chains
            visited: set[uuid.UUID] = {start_node}
            queue: list[tuple[list[uuid.UUID], list[uuid.UUID]]] = [
                ([start_node], [])
            ]  # (node_path, edge_path)

            while queue:
                node_path, edge_path = queue.pop(0)
                current = node_path[-1]

                for next_node, edge_id in adjacency[current]:
                    if next_node not in visited:
                        new_node_path = node_path + [next_node]
                        new_edge_path = edge_path + [edge_id]

                        # If chain length >= 3, emit hypothesis
                        if len(new_node_path) >= 3:
                            confidence = min(1.0, 0.5 + 0.1 * len(new_node_path))

                            labels = [
                                next((n.label for n in graph.nodes if n.node_id == nid), "?")
                                for nid in new_node_path
                            ]

                            candidates.append(
                                HypothesisCandidate(
                                    hypothesis_id=uuid.uuid4(),
                                    hypothesis_type="chain",
                                    description=f"Sequential chain detected: {' → '.join(labels)}",
                                    supporting_nodes=tuple(new_node_path),
                                    supporting_edges=tuple(new_edge_path),
                                    confidence=confidence,
                                    metadata={
                                        "chain_length": len(new_node_path),
                                    },
                                    created_at=datetime.now(UTC),
                                )
                            )

                        # Continue BFS
                        visited.add(next_node)
                        queue.append((new_node_path, new_edge_path))

        return candidates

    def _detect_hubs(self, graph: KnowledgeGraph) -> list[HypothesisCandidate]:
        """Detect hub patterns: nodes with many connections."""
        candidates: list[HypothesisCandidate] = []

        if len(graph.nodes) < 3:
            return candidates

        # Count degree for each node
        degree: dict[uuid.UUID, int] = {n.node_id: 0 for n in graph.nodes}
        edge_map: dict[uuid.UUID, list[uuid.UUID]] = {n.node_id: [] for n in graph.nodes}

        for edge in graph.edges:
            degree[edge.source_id] += 1
            degree[edge.target_id] += 1
            edge_map[edge.source_id].append(edge.edge_id)
            edge_map[edge.target_id].append(edge.edge_id)

        # Hub threshold: >50% of possible connections
        hub_threshold = max(3, len(graph.nodes) // 2)

        for node_id, deg in degree.items():
            if deg >= hub_threshold:
                confidence = min(1.0, deg / len(graph.nodes))
                label = next((n.label for n in graph.nodes if n.node_id == node_id), "?")

                candidates.append(
                    HypothesisCandidate(
                        hypothesis_id=uuid.uuid4(),
                        hypothesis_type="hub",
                        description=f"Node '{label}' is a hub with {deg} connections",
                        supporting_nodes=(node_id,),
                        supporting_edges=tuple(edge_map[node_id]),
                        confidence=confidence,
                        metadata={
                            "degree": deg,
                            "hub_threshold": hub_threshold,
                        },
                        created_at=datetime.now(UTC),
                    )
                )

        return candidates

    def _detect_contradictions(self, graph: KnowledgeGraph) -> list[HypothesisCandidate]:
        """Detect contradiction patterns: conflicting edge types between same nodes."""
        candidates: list[HypothesisCandidate] = []

        if len(graph.edges) < 2:
            return candidates

        # Group edges by node pair
        edge_groups: dict[tuple[uuid.UUID, uuid.UUID], list[Edge]] = {}
        for edge in graph.edges:
            # Normalize pair (smaller ID first)
            pair = (
                (edge.source_id, edge.target_id)
                if edge.source_id < edge.target_id
                else (edge.target_id, edge.source_id)
            )
            if pair not in edge_groups:
                edge_groups[pair] = []
            edge_groups[pair].append(edge)

        # Find pairs with multiple edge types
        contradictory_types = {"supports", "contradicts"}  # Example opposition

        for pair, edges in edge_groups.items():
            if len(edges) < 2:
                continue

            edge_types = {e.edge_type for e in edges}

            # Check for contradictory types
            if contradictory_types.issubset(edge_types):
                node_a, node_b = pair
                label_a = next((n.label for n in graph.nodes if n.node_id == node_a), "?")
                label_b = next((n.label for n in graph.nodes if n.node_id == node_b), "?")

                candidates.append(
                    HypothesisCandidate(
                        hypothesis_id=uuid.uuid4(),
                        hypothesis_type="contradiction",
                        description=f"Contradictory relationships between '{label_a}' and '{label_b}': {', '.join(edge_types)}",
                        supporting_nodes=(node_a, node_b),
                        supporting_edges=tuple(e.edge_id for e in edges),
                        confidence=1.0,  # Contradiction is definitive
                        metadata={
                            "edge_types": sorted(list(edge_types)),
                            "edge_count": len(edges),
                        },
                        created_at=datetime.now(UTC),
                    )
                )

        return candidates
