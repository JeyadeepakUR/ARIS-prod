"""
Policy-bounded Research Planner (Module 9) for ARIS.

Deterministic planning over knowledge graphs that proposes next research
actions without executing them and without mutating inputs.

No external dependencies. Strict type hints. Evidence-backed actions.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass
from typing import Iterable

from aris.knowledge_graph import Edge, KnowledgeGraph


@dataclass(frozen=True)
class ResearchAction:
    """
    Immutable, evidence-backed research action proposal.

    Attributes:
        action_id: Deterministic UUID derived from graph and content
        action_type: One of: investigate_gap, resolve_contradiction, strengthen_evidence
        description: Human-readable action description
        evidence: Concrete evidence motivating the action
        rationale: Clear rationale explaining why this action is useful
        priority: Priority score in [0.0, 1.0]
        metadata: Additional structured details (no IDs required)
    """

    action_id: uuid.UUID
    action_type: str
    description: str
    evidence: str
    rationale: str
    priority: float
    metadata: dict[str, str]

    def __post_init__(self) -> None:  # type: ignore[override]
        # Enforce bounds and completeness deterministically
        if not 0.0 <= self.priority <= 1.0:
            raise ValueError(f"priority must be within [0.0, 1.0], got {self.priority}")
        if not self.evidence.strip():
            raise ValueError("evidence must be non-empty")
        if not self.rationale.strip():
            raise ValueError("rationale must be non-empty")


@dataclass(frozen=True)
class PlannerContext:
    """
    Immutable context for planning.

    Attributes:
        graph: Source knowledge graph (not mutated)
        strategy: Planning strategy: gap-driven | contradiction-driven | weak-evidence
        max_actions: Upper bound on actions to return
    """

    graph: KnowledgeGraph
    strategy: str
    max_actions: int = 10


class ResearchPlanner:
    """
    Deterministic research planner that proposes next actions.

    Constraints:
    - Does not execute actions
    - Does not mutate the graph or traces
    - No autonomy loops or implicit learning/optimization
    """

    # Relationship type hints for contradiction detection (explicit, deterministic)
    _SUPPORT_TYPES: set[str] = {"supports", "agrees", "confirms"}
    _CONTRADICT_TYPES: set[str] = {"contradicts", "refutes", "disagrees"}

    def plan(self, context: PlannerContext) -> list[ResearchAction]:
        strategy = context.strategy.strip().lower()

        if strategy == "gap-driven":
            actions = self._plan_gap_driven(context.graph)
        elif strategy == "contradiction-driven":
            actions = self._plan_contradictions(context.graph)
        elif strategy in ("weak-evidence", "weak-evidence refinement", "weak-evidence-refinement"):
            actions = self._plan_weak_evidence(context.graph)
        else:
            raise ValueError(
                "Unsupported strategy: "
                + strategy
                + ". Supported: gap-driven, contradiction-driven, weak-evidence refinement"
            )

        # Deterministic ranking: priority desc, then action_type, description, action_id
        actions_sorted = sorted(
            actions,
            key=lambda a: (
                -a.priority,
                a.action_type,
                a.description,
                a.action_id.hex,
            ),
        )

        return actions_sorted[: max(0, context.max_actions)]

    # ----------------------------
    # Strategy: Gap-Driven
    # ----------------------------
    def _plan_gap_driven(self, graph: KnowledgeGraph) -> list[ResearchAction]:
        # Compute degree for each node/document identifier that appears in the graph
        ids = self._all_identifiers(graph)
        degrees: dict[uuid.UUID, int] = {i: 0 for i in ids}
        for e in graph.edges:
            if e.source_id in degrees:
                degrees[e.source_id] += 1
            if e.target_id in degrees:
                degrees[e.target_id] += 1

        actions: list[ResearchAction] = []
        for node in graph.nodes:
            # Prefer to evaluate by document_id to be robust to different edge conventions
            deg_doc = degrees.get(node.document_id, 0)
            deg_node = degrees.get(node.node_id, 0)
            degree = max(deg_doc, deg_node)

            if degree <= 1:
                # Priority: 1.0 for isolated, 0.8 for degree 1, 0.6 for degree 2+, clamped
                base = 1.0 if degree == 0 else 0.8
                priority = max(0.0, min(1.0, base))

                evidence = f"Document '{node.label}' degree={degree} in graph {graph.graph_id}"
                rationale = (
                    "Low connectivity suggests missing links or unintegrated knowledge; "
                    "investigate related documents or references."
                )
                description = f"Investigate connections for document '{node.label}'"
                action_id = self._deterministic_id(
                    graph.graph_id,
                    f"gap:{node.document_id}:{node.node_id}:{degree}",
                )
                actions.append(
                    ResearchAction(
                        action_id=action_id,
                        action_type="investigate_gap",
                        description=description,
                        evidence=evidence,
                        rationale=rationale,
                        priority=priority,
                        metadata={
                            "document_id": str(node.document_id),
                            "node_id": str(node.node_id),
                            "degree": str(degree),
                        },
                    )
                )

        return actions

    # ----------------------------
    # Strategy: Contradiction-Driven
    # ----------------------------
    def _plan_contradictions(self, graph: KnowledgeGraph) -> list[ResearchAction]:
        # Group edge types by unordered pair
        pair_types: dict[tuple[uuid.UUID, uuid.UUID], set[str]] = {}
        for e in graph.edges:
            pair = self._pair_key(e)
            if pair not in pair_types:
                pair_types[pair] = set()
            pair_types[pair].add(e.edge_type.lower())

        actions: list[ResearchAction] = []
        for pair, types in pair_types.items():
            if not types:
                continue
            # Determine if support and contradiction types co-occur
            has_support = any(t in self._SUPPORT_TYPES for t in types)
            has_contra = any(t in self._CONTRADICT_TYPES for t in types)
            if has_support and has_contra:
                s, t = pair
                evidence = (
                    f"Edges between {s} and {t} contain both supportive and refutational types: "
                    + ", ".join(sorted(types))
                )
                rationale = (
                    "Conflicting relations indicate potential contradiction; collect primary sources "
                    "and adjudicate the claim."
                )
                description = "Resolve contradiction between linked documents"
                # Priority scaled by diversity of types (more disagreement → higher)
                diversity = min(1.0, len(types) / 4.0)
                priority = max(0.0, min(1.0, 0.8 + 0.2 * diversity))
                action_id = self._deterministic_id(
                    graph.graph_id,
                    f"contradiction:{s}:{t}:{','.join(sorted(types))}",
                )
                actions.append(
                    ResearchAction(
                        action_id=action_id,
                        action_type="resolve_contradiction",
                        description=description,
                        evidence=evidence,
                        rationale=rationale,
                        priority=priority,
                        metadata={
                            "pair_source": str(s),
                            "pair_target": str(t),
                            "types": ",".join(sorted(types)),
                        },
                    )
                )

        return actions

    # ----------------------------
    # Strategy: Weak-Evidence Refinement
    # ----------------------------
    def _plan_weak_evidence(self, graph: KnowledgeGraph) -> list[ResearchAction]:
        actions: list[ResearchAction] = []
        for e in graph.edges:
            evidence_len = len(e.evidence.strip())
            # Weak if confidence < 0.6 or very short textual evidence
            is_weak = e.confidence < 0.6 or evidence_len < 40
            if not is_weak:
                continue

            # Priority: stronger when confidence is lower and evidence is shorter
            conf_component = max(0.0, min(1.0, (0.6 - e.confidence) / 0.6))
            len_component = 1.0 if evidence_len == 0 else max(0.0, min(1.0, 40.0 / (evidence_len + 40.0)))
            priority = max(0.0, min(1.0, 0.5 * conf_component + 0.5 * len_component))

            evidence = (
                f"Edge {e.edge_id} has confidence={e.confidence:.2f} and evidence_len={evidence_len}"
            )
            rationale = (
                "Weak justification detected; refine by gathering additional citations, "
                "cross-document references, or more detailed quotations."
            )
            description = f"Strengthen evidence for relation '{e.edge_type}'"
            action_id = self._deterministic_id(
                graph.graph_id,
                f"weak:{e.edge_id}:{e.source_id}:{e.target_id}:{e.edge_type}:{e.confidence}:{evidence_len}",
            )
            actions.append(
                ResearchAction(
                    action_id=action_id,
                    action_type="strengthen_evidence",
                    description=description,
                    evidence=evidence,
                    rationale=rationale,
                    priority=priority,
                    metadata={
                        "edge_id": str(e.edge_id),
                        "edge_type": e.edge_type,
                        "source_id": str(e.source_id),
                        "target_id": str(e.target_id),
                    },
                )
            )

        return actions

    # ----------------------------
    # Utilities
    # ----------------------------
    def _all_identifiers(self, graph: KnowledgeGraph) -> set[uuid.UUID]:
        ids: set[uuid.UUID] = set()
        for n in graph.nodes:
            ids.add(n.node_id)
            ids.add(n.document_id)
        for e in graph.edges:
            ids.add(e.source_id)
            ids.add(e.target_id)
        return ids

    def _pair_key(self, e: Edge) -> tuple[uuid.UUID, uuid.UUID]:
        a, b = e.source_id, e.target_id
        return (a, b) if a.int <= b.int else (b, a)

    def _deterministic_id(self, graph_id: uuid.UUID, name: str) -> uuid.UUID:
        # Use a stable namespace derived from the graph ID to avoid collisions across graphs
        ns = uuid.uuid5(uuid.NAMESPACE_URL, f"aris.research-planner/{graph_id}")
        return uuid.uuid5(ns, name)
