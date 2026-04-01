"""Phase 1 Graph: Document linking and policy-bounded research planning (Modules 7-9).

This package provides deterministic document ingestion, cross-document linking,
and policy-bounded research planning without ML dependencies:
- Document ingestion and corpus canonicalization
- Knowledge graph construction with evidence-backed edges
- Deterministic research planning (gap-driven, contradiction-driven, weak-evidence)
- Hypothesis induction from graph patterns (Module 13)
- Hypothesis impact scoring using entity/relation signals (Module 13 enhancement)

All operations are reproducible, falsifiable, and non-mutating.
"""

from aris.graph.document_ingestion import (
    Document,
    DocumentCorpus,
    DocumentIngestor,
    PDFLoader,
    PlainTextLoader,
)
from aris.graph.hypothesis_induction import (
    HypothesisCandidate,
    HypothesisInductionEngine,
)
from aris.graph.hypothesis_impact import (
    HypothesisImpactScore,
    HypothesisImpactScoringTool,
)
from aris.graph.bridge_discovery import BridgeCandidate, BridgeDiscoveryEngine
from aris.graph.contradiction_engine import ContradictionEngine, ContradictionRecord
from aris.graph.hypothesis_portfolio import HypothesisPortfolioEngine, ResearchHypothesis
from aris.graph.knowledge_graph import (
    Edge,
    KnowledgeGraph,
    LinkCandidate,
    LinkMaterializer,
    Linker,
    Node,
    add_edges_to_graph,
    build_graph_from_corpus,
)
from aris.graph.research_planner import PlannerContext, ResearchAction, ResearchPlanner

__all__ = [
    "Document",
    "DocumentCorpus",
    "PlainTextLoader",
    "PDFLoader",
    "DocumentIngestor",
    "Node",
    "Edge",
    "KnowledgeGraph",
    "LinkCandidate",
    "Linker",
    "LinkMaterializer",
    "build_graph_from_corpus",
    "add_edges_to_graph",
    "HypothesisCandidate",
    "HypothesisInductionEngine",
    "HypothesisImpactScore",
    "HypothesisImpactScoringTool",
    "ContradictionRecord",
    "ContradictionEngine",
    "BridgeCandidate",
    "BridgeDiscoveryEngine",
    "ResearchHypothesis",
    "HypothesisPortfolioEngine",
    "ResearchAction",
    "PlannerContext",
    "ResearchPlanner",
]
