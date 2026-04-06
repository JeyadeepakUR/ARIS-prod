"""FastAPI application for ARIS research intelligence service."""

from aris import __version__
from aris.core.semantic_analyzer import SemanticAnalyzer
from aris.graph.bridge_discovery import BridgeDiscoveryEngine
from aris.graph.contradiction_engine import ContradictionEngine
from aris.graph.hypothesis_portfolio import HypothesisPortfolioEngine


def create_app():
    """Create FastAPI application instance.

    FastAPI import is intentionally local so core ARIS usage remains dependency-free
    unless API mode is explicitly enabled.
    """

    try:
        from fastapi import Body, FastAPI
    except ImportError as exc:  # pragma: no cover
        raise ImportError(
            "FastAPI dependencies are missing. Install with: pip install aris[api]"
        ) from exc

    from aris.api.schemas import AnalyzeRequest, AnalyzeResponse, HealthResponse

    app = FastAPI(
        title="ARIS Research Intelligence API",
        version=__version__,
        description=(
            "Cross-paper hypothesis generation, contradiction mining, and cross-domain "
            "bridge discovery for innovation workflows."
        ),
    )

    semantic_analyzer = SemanticAnalyzer()
    contradiction_engine = ContradictionEngine()
    bridge_engine = BridgeDiscoveryEngine()
    portfolio_engine = HypothesisPortfolioEngine()

    @app.get("/v1/health", response_model=HealthResponse)
    def health() -> HealthResponse:
        return HealthResponse(version=__version__)

    @app.post("/v1/analyze", response_model=AnalyzeResponse)
    def analyze(payload: AnalyzeRequest = Body(...)) -> AnalyzeResponse:
        profiles = [
            semantic_analyzer.analyze(
                item.text,
                document_id=item.document_id,
                domain=item.domain,
            )
            for item in payload.documents
        ]

        contradictions = contradiction_engine.find_contradictions(profiles)
        bridges = bridge_engine.discover(profiles, top_k=payload.top_k_bridges)
        hypotheses = portfolio_engine.generate(
            profiles,
            contradictions,
            bridges,
            max_items=payload.max_hypotheses,
        )

        return AnalyzeResponse(
            profile_count=len(profiles),
            contradiction_count=len(contradictions),
            bridge_count=len(bridges),
            hypothesis_count=len(hypotheses),
            contradictions=[
                {
                    "document_a": item.document_a,
                    "document_b": item.document_b,
                    "claim_a": item.claim_a,
                    "claim_b": item.claim_b,
                    "overlap_terms": list(item.overlap_terms),
                    "severity": item.severity,
                    "rationale": item.rationale,
                }
                for item in contradictions
            ],
            bridges=[
                {
                    "source_domain": item.source_domain,
                    "target_domain": item.target_domain,
                    "bridge_concept": item.bridge_concept,
                    "shared_keywords": list(item.shared_keywords),
                    "novelty": item.novelty,
                    "rationale": item.rationale,
                }
                for item in bridges
            ],
            hypotheses=[
                {
                    "hypothesis_id": item.hypothesis_id,
                    "mode": item.mode,
                    "hypothesis_type": item.hypothesis_type,
                    "statement": item.statement,
                    "evidence_documents": list(item.evidence_documents),
                    "novelty_score": item.novelty_score,
                    "contradiction_leverage": item.contradiction_leverage,
                    "feasibility_score": item.feasibility_score,
                    "overall_priority": item.overall_priority,
                }
                for item in hypotheses
            ],
        )

    return app


app = create_app()
