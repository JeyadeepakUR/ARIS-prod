"""Hypothesis portfolio generation for deep and wide research exploration."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass

from aris.core.semantic_analyzer import SemanticProfile
from aris.graph.bridge_discovery import BridgeCandidate
from aris.graph.contradiction_engine import ContradictionRecord


@dataclass(frozen=True)
class ResearchHypothesis:
    """Actionable hypothesis candidate with multi-objective scores."""

    hypothesis_id: str
    mode: str
    hypothesis_type: str
    statement: str
    evidence_documents: tuple[str, ...]
    novelty_score: float
    contradiction_leverage: float
    feasibility_score: float
    overall_priority: float


class HypothesisPortfolioEngine:
    """Generate prioritized deep and wide hypotheses from profiles and signals."""

    def generate(
        self,
        profiles: list[SemanticProfile],
        contradictions: list[ContradictionRecord],
        bridges: list[BridgeCandidate],
        *,
        max_items: int = 30,
    ) -> list[ResearchHypothesis]:
        hypotheses: list[ResearchHypothesis] = []
        hypotheses.extend(self._deep_hypotheses(profiles, contradictions))
        hypotheses.extend(self._wide_hypotheses(bridges))

        ranked = sorted(
            hypotheses,
            key=lambda hyp: (-hyp.overall_priority, hyp.mode, hyp.hypothesis_id),
        )
        return ranked[: max(1, max_items)]

    def _deep_hypotheses(
        self,
        profiles: list[SemanticProfile],
        contradictions: list[ContradictionRecord],
    ) -> list[ResearchHypothesis]:
        items: list[ResearchHypothesis] = []

        for contradiction in contradictions:
            statement = (
                "Condition-specific effect may reconcile opposing claims across "
                f"{contradiction.document_a} and {contradiction.document_b}."
            )
            evidence_docs = (contradiction.document_a, contradiction.document_b)
            novelty = min(1.0, 0.5 + contradiction.severity * 0.4)
            leverage = contradiction.severity
            feasibility = 0.6
            priority = round((0.45 * novelty) + (0.4 * leverage) + (0.15 * feasibility), 3)
            items.append(
                ResearchHypothesis(
                    hypothesis_id=self._make_id("deep", statement),
                    mode="deep",
                    hypothesis_type="contradiction_resolution",
                    statement=statement,
                    evidence_documents=evidence_docs,
                    novelty_score=round(novelty, 3),
                    contradiction_leverage=round(leverage, 3),
                    feasibility_score=round(feasibility, 3),
                    overall_priority=priority,
                )
            )

        if items:
            return items

        for profile in profiles:
            if not profile.claims:
                continue
            top_claim = profile.claims[0]
            statement = (
                "Focused replication and ablation study should validate claim under "
                f"boundary conditions in {profile.document_id}."
            )
            novelty = 0.45
            leverage = 0.2
            feasibility = min(1.0, 0.4 + top_claim.confidence)
            priority = round((0.45 * novelty) + (0.4 * leverage) + (0.15 * feasibility), 3)
            items.append(
                ResearchHypothesis(
                    hypothesis_id=self._make_id("deep", statement),
                    mode="deep",
                    hypothesis_type="intra_domain_refinement",
                    statement=statement,
                    evidence_documents=(profile.document_id,),
                    novelty_score=round(novelty, 3),
                    contradiction_leverage=round(leverage, 3),
                    feasibility_score=round(feasibility, 3),
                    overall_priority=priority,
                )
            )

        return items

    def _wide_hypotheses(self, bridges: list[BridgeCandidate]) -> list[ResearchHypothesis]:
        items: list[ResearchHypothesis] = []

        for bridge in bridges:
            statement = (
                f"Transferring '{bridge.bridge_concept}' from {bridge.source_domain} "
                f"to {bridge.target_domain} may unlock unexplored performance regimes."
            )
            novelty = min(1.0, 0.55 + bridge.novelty * 0.35)
            leverage = 0.35
            feasibility = 0.5
            priority = round((0.45 * novelty) + (0.4 * leverage) + (0.15 * feasibility), 3)
            items.append(
                ResearchHypothesis(
                    hypothesis_id=self._make_id("wide", statement),
                    mode="wide",
                    hypothesis_type="cross_domain_transfer",
                    statement=statement,
                    evidence_documents=(bridge.source_domain, bridge.target_domain),
                    novelty_score=round(novelty, 3),
                    contradiction_leverage=round(leverage, 3),
                    feasibility_score=round(feasibility, 3),
                    overall_priority=priority,
                )
            )

        return items

    def _make_id(self, prefix: str, statement: str) -> str:
        digest = hashlib.sha256(f"{prefix}:{statement}".encode("utf-8")).hexdigest()
        return f"hyp_{digest[:16]}"
