from __future__ import annotations

from typing import TypedDict, Annotated
import operator


class ConceptNode(TypedDict):
    node_id: str
    label: str
    domain: str
    source_chunk_ids: list[str]
    confidence: float


class IntraEdge(TypedDict):
    source_node_id: str
    target_node_id: str
    edge_type: str          # supports | contradicts | applies | extends | uses
    evidence: str
    confidence: float


class BridgeEdge(TypedDict):
    source_node_id: str
    target_node_id: str
    source_domain: str
    target_domain: str
    bridge_concept: str     # the transfer mechanism
    evidence_a: str         # evidence from source domain
    evidence_b: str         # evidence from target domain
    cosine_similarity: float
    confidence: float
    is_multi_hop: bool
    intermediate_node_ids: list[str]  # empty for direct bridges


class Contradiction(TypedDict):
    claim_a_text: str
    claim_b_text: str
    claim_a_chunk_id: str
    claim_b_chunk_id: str
    author_a: str
    author_b: str
    contradiction_type: str  # direct | methodological | scope
    severity: float
    llm_reasoning: str


class Hypothesis(TypedDict):
    bridge_edge_source: str
    bridge_edge_target: str
    statement: str
    null_hypothesis: str
    hypothesis_type: str     # causal | correlational | technology_transfer | mechanistic
    methodology_hint: str
    testability_score: float
    novelty: str             # novel | known | uncertain
    supporting_evidence: list[str]
    confidence: float


class ResearchGap(TypedDict):
    node_a_id: str
    node_b_id: str
    common_neighbor_ids: list[str]
    gap_description: str
    investigation_priority: float
    rationale: str


class StreamEvent(TypedDict):
    agent_node: str
    event_type: str
    # concept_found | bridge_found | contradiction_found | hypothesis_generated
    # gap_identified | step_complete | run_complete | error
    content: dict
    timestamp: str


class ResearchState(TypedDict):
    # ── Identifiers ──────────────────────────────────────────────────────────
    graph_id: str
    workspace_id: str
    document_ids: list[str]
    thread_id: str

    # ── Accumulated outputs (append-only via operator.add) ───────────────────
    concepts: Annotated[list[ConceptNode], operator.add]
    intra_edges: Annotated[list[IntraEdge], operator.add]
    bridge_candidates: Annotated[list[BridgeEdge], operator.add]
    bridges: Annotated[list[BridgeEdge], operator.add]
    contradictions: Annotated[list[Contradiction], operator.add]
    hypotheses: Annotated[list[Hypothesis], operator.add]
    gaps: Annotated[list[ResearchGap], operator.add]
    stream_events: Annotated[list[StreamEvent], operator.add]

    # ── Orchestrator control ─────────────────────────────────────────────────
    orchestrator_plan: list[str]   # ordered list of agent nodes to run
    completed_steps: Annotated[list[str], operator.add]
    iteration: int
    error: str | None

    # ── Routing (set by orchestrator, consumed by route_next) ───────────────
    _next_node: str

    # ── Human-in-the-loop ────────────────────────────────────────────────────
    human_feedback: str | None     # researcher can inject direction mid-run
