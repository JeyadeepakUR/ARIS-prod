"""Tests for Module 9: Research Planner."""

from __future__ import annotations

import uuid as uuid_module
from datetime import UTC, datetime

from aris.document_ingestion import Document, DocumentCorpus
from aris.knowledge_graph import Edge, KnowledgeGraph, Node, add_edges_to_graph, build_graph_from_corpus
from aris.research_planner import PlannerContext, ResearchAction, ResearchPlanner


def make_doc(content: str, source: str, doc_id_num: int) -> Document:
    return Document(
        content=content,
        source=source,
        format="txt",
        metadata={},
        document_id=uuid_module.UUID(f"00000000-0000-0000-0000-{doc_id_num:012d}"),
        created_at=datetime.now(UTC),
    )


def sample_graph_for_planner() -> KnowledgeGraph:
    corpus = DocumentCorpus(
        documents=[
            make_doc("Alpha content", "alpha.txt", 1),
            make_doc("Beta content", "beta.txt", 2),
            make_doc("Gamma content", "gamma.txt", 3),
        ],
        corpus_id=uuid_module.uuid4(),
        created_at=datetime.now(UTC),
    )
    base = build_graph_from_corpus(corpus)

    # Create edges: two between doc1 and doc2 with opposing semantics, and one weak edge
    doc1 = uuid_module.UUID("00000000-0000-0000-0000-000000000001")
    doc2 = uuid_module.UUID("00000000-0000-0000-0000-000000000002")

    e1 = Edge(
        edge_id=uuid_module.uuid4(),
        source_id=doc1,
        target_id=doc2,
        edge_type="supports",
        evidence="Beta cites Alpha's method.",
        reasoning_trace_id=uuid_module.uuid4(),
        confidence=0.75,
        metadata={},
        created_at=datetime.now(UTC),
    )
    e2 = Edge(
        edge_id=uuid_module.uuid4(),
        source_id=doc2,
        target_id=doc1,
        edge_type="refutes",
        evidence="Counterexample contradicts stated claim.",
        reasoning_trace_id=uuid_module.uuid4(),
        confidence=0.9,
        metadata={},
        created_at=datetime.now(UTC),
    )
    e3 = Edge(
        edge_id=uuid_module.uuid4(),
        source_id=doc1,
        target_id=doc1,
        edge_type="mentions",
        evidence="weak",  # very short ==> weak evidence
        reasoning_trace_id=uuid_module.uuid4(),
        confidence=0.55,  # below threshold
        metadata={},
        created_at=datetime.now(UTC),
    )

    return add_edges_to_graph(base, [e1, e2, e3])


def test_gap_driven_generates_actions():
    graph = sample_graph_for_planner()
    # Node 3 should be isolated (no edges referencing doc3)
    planner = ResearchPlanner()
    ctx = PlannerContext(graph=graph, strategy="gap-driven", max_actions=10)
    actions = planner.plan(ctx)

    assert len(actions) >= 1
    for a in actions:
        assert isinstance(a, ResearchAction)
        assert a.action_type in {"investigate_gap", "resolve_contradiction", "strengthen_evidence"}
        assert 0.0 <= a.priority <= 1.0
        assert a.evidence
        assert a.rationale

    # Ensure at least one action targets a gap (description contains 'Investigate connections')
    assert any("Investigate connections" in a.description for a in actions)


def test_contradiction_driven_generates_actions():
    graph = sample_graph_for_planner()
    planner = ResearchPlanner()
    ctx = PlannerContext(graph=graph, strategy="contradiction-driven", max_actions=10)
    actions = planner.plan(ctx)

    assert any(a.action_type == "resolve_contradiction" for a in actions)
    # Evidence should mention both supportive and refutational types
    for a in actions:
        if a.action_type == "resolve_contradiction":
            assert "supports" in a.evidence or "refutes" in a.evidence


def test_weak_evidence_refinement_generates_actions():
    graph = sample_graph_for_planner()
    planner = ResearchPlanner()
    ctx = PlannerContext(graph=graph, strategy="weak-evidence refinement", max_actions=10)
    actions = planner.plan(ctx)
    assert any(a.action_type == "strengthen_evidence" for a in actions)


def test_determinism_and_ranking_stability():
    graph = sample_graph_for_planner()
    planner = ResearchPlanner()

    ctx_gap = PlannerContext(graph=graph, strategy="gap-driven", max_actions=50)
    ctx_con = PlannerContext(graph=graph, strategy="contradiction-driven", max_actions=50)
    ctx_weak = PlannerContext(graph=graph, strategy="weak-evidence", max_actions=50)

    r1 = planner.plan(ctx_gap) + planner.plan(ctx_con) + planner.plan(ctx_weak)
    r2 = planner.plan(ctx_gap) + planner.plan(ctx_con) + planner.plan(ctx_weak)

    # Exact equality of serialized views (excluding object identity)
    def view(actions: list[ResearchAction]) -> list[tuple[str, str, str, str, float, tuple[tuple[str, str], ...]]]:
        return [
            (
                a.action_type,
                a.description,
                a.evidence,
                a.rationale,
                a.priority,
                tuple(sorted(a.metadata.items())),
            )
            for a in actions
        ]

    assert view(r1) == view(r2)

    # Ensure sorted by priority, then secondary keys are stable
    priorities = [a.priority for a in r1]
    assert priorities == sorted(priorities, reverse=True)


def test_planner_context_immutable():
    graph = sample_graph_for_planner()
    ctx = PlannerContext(graph=graph, strategy="gap-driven", max_actions=10)
    # Attempting to mutate should fail due to frozen dataclass
    try:
        # type: ignore[attr-defined]
        ctx.strategy = "weak-evidence"
        mutated = True
    except Exception:
        mutated = False
    assert mutated is False
