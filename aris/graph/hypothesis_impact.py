"""Hypothesis Impact Scoring (Module 13 Enhancement).

Ranks hypotheses by research potential using signals from:
- Entity mention frequency (Module 10B: cross-document mentions)
- Research relations (Module 12: ACHIEVES, OUTPERFORMS, IMPROVES, SUPPORTS)
- Citation networks (relation connectivity)
- Semantic impact (hypothesis supports novel relations)

Novel contribution: Synthesizes entity extraction + relation induction
to produce research-value-aware hypothesis scoring.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

from aris.core.tool import Tool
from aris.graph.hypothesis_induction import HypothesisCandidate


@dataclass(frozen=True)
class HypothesisImpactScore:
    """Immutable hypothesis impact assessment.

    Attributes:
        hypothesis_id: Reference to HypothesisCandidate
        hypothesis_type: Type (gap, cluster, chain, hub, contradiction)
        impact_score: Overall research value (0-1)
        novelty_signal: Hypothesis introduces new connections (0-1)
        citation_signal: Hypothesis involves frequently-mentioned entities (0-1)
        relation_signal: Hypothesis connects research-relation nodes (0-1)
        base_confidence: Original hypothesis confidence
        total_signal: Sum of normalized signals before scaling
        provenance: Scoring metadata (weights, contributing factors)
    """

    hypothesis_id: str
    hypothesis_type: str
    impact_score: float
    novelty_signal: float
    citation_signal: float
    relation_signal: float
    base_confidence: float
    total_signal: float
    provenance: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return {
            "hypothesis_id": self.hypothesis_id,
            "hypothesis_type": self.hypothesis_type,
            "impact_score": float(max(0.0, min(1.0, self.impact_score))),
            "novelty_signal": float(max(0.0, min(1.0, self.novelty_signal))),
            "citation_signal": float(max(0.0, min(1.0, self.citation_signal))),
            "relation_signal": float(max(0.0, min(1.0, self.relation_signal))),
            "base_confidence": float(max(0.0, min(1.0, self.base_confidence))),
            "total_signal": float(self.total_signal),
            "provenance": self.provenance,
        }


class HypothesisImpactScoringTool(Tool):
    """Rank hypotheses by research impact using extracted signals.

    Input schema (JSON string):
        {
          "hypotheses": [
            {"hypothesis_id": "uuid", "hypothesis_type": "gap|cluster|chain|hub|contradiction",
             "description": "...", "confidence": 0.0-1.0, "supporting_nodes": [...]},
            ...
          ],
          "entity_mentions": {
            "entity_span": {
              "mention_count": int,  # Cross-document mention frequency
              "entity_type": "METHOD|DATASET|METRIC|..."
            },
            ...
          },
          "relations": [
            {"relation_type": "ACHIEVES|OUTPERFORMS|IMPROVES|SUPPORTS",
             "subject_span": "...", "object_span": "...", "confidence": 0.0-1.0},
            ...
          ]
        }

    Output: JSON list of HypothesisImpactScore dicts, sorted by impact_score descending.
    """

    # Relation weights: higher for novel research outcomes
    RELATION_TYPE_WEIGHTS = {
        "ACHIEVES": 0.9,  # Achievement metrics → high novelty
        "OUTPERFORMS": 1.0,  # Comparative claims → highest novelty
        "IMPROVES": 0.85,  # Improvement claims
        "SUPPORTS": 0.75,  # Support/validation
        "CAUSES": 0.8,  # Causal claims
        "RELATED_TO": 0.5,  # Generic relation
        "USES": 0.6,  # Methodology relation
        "DEPENDS_ON": 0.65,  # Dependency relation
        "CONTRADICTS": 0.95,  # Contradictions → high novelty
    }

    # Hypothesis type base weights
    HYPOTHESIS_TYPE_WEIGHTS = {
        "gap": 0.9,  # Gaps are novel (missing connections)
        "chain": 0.75,  # Chains are less novel (linear connections)
        "cluster": 0.7,  # Clusters are established structures
        "hub": 0.6,  # Hubs are known high-degree nodes
        "contradiction": 1.0,  # Contradictions are most impactful
    }

    def __init__(self) -> None:
        pass

    @property
    def name(self) -> str:
        return "hypothesis_impact_scoring"

    def execute(self, input_text: str) -> str:
        text = input_text or ""
        if not text.strip():
            return "[]"

        try:
            data = self._parse_input(text)
            snapshot = json.dumps(data, sort_keys=True)
            scored = self._score_hypotheses(data)
            # Ensure immutability
            assert snapshot == json.dumps(data, sort_keys=True)
            scored.sort(key=lambda h: h.impact_score, reverse=True)
            return json.dumps([s.to_dict() for s in scored], ensure_ascii=True, sort_keys=True)
        except Exception as exc:  # noqa: BLE001
            return json.dumps(
                {
                    "error": str(exc),
                    "error_type": exc.__class__.__name__,
                    "module": "hypothesis_impact",
                },
                ensure_ascii=True,
                sort_keys=True,
            )

    def _parse_input(self, text: str) -> dict[str, Any]:
        data = json.loads(text)
        if not isinstance(data, dict):
            raise ValueError("Input must be a JSON object")
        return data

    def _score_hypotheses(self, data: dict[str, Any]) -> list[HypothesisImpactScore]:
        hypotheses = data.get("hypotheses", [])
        entity_mentions = data.get("entity_mentions", {})
        relations = data.get("relations", [])

        if not hypotheses:
            return []

        # Index relations by involved entities for faster lookup
        relation_entities: set[str] = set()
        relation_dicts = []
        for rel in relations:
            if isinstance(rel, dict):
                relation_dicts.append(rel)
                relation_entities.add(rel.get("subject_span", ""))
                relation_entities.add(rel.get("object_span", ""))

        scores: list[HypothesisImpactScore] = []

        for hyp in hypotheses:
            if not isinstance(hyp, dict):
                continue

            hyp_id = hyp.get("hypothesis_id", "unknown")
            hyp_type = hyp.get("hypothesis_type", "unknown")
            base_conf = hyp.get("confidence", 0.5)
            desc = hyp.get("description", "")

            # Score components
            novelty_sig = self._compute_novelty_signal(hyp_type, desc, entity_mentions)
            citation_sig = self._compute_citation_signal(desc, entity_mentions)
            relation_sig = self._compute_relation_signal(desc, relation_dicts)

            # Aggregate signals
            type_weight = self.HYPOTHESIS_TYPE_WEIGHTS.get(hyp_type, 0.5)
            total_sig = novelty_sig + citation_sig + relation_sig

            # Combine: base_confidence provides anchor, signals provide novelty boost
            impact = base_conf * type_weight + 0.3 * (total_sig / 3.0)
            impact = min(1.0, max(0.0, impact))

            provenance = {
                "hypothesis_type": hyp_type,
                "type_weight": type_weight,
                "base_confidence": base_conf,
                "novelty_components": {
                    "has_gap_keywords": "gap" in desc.lower(),
                    "has_novel_keywords": any(
                        kw in desc.lower() for kw in ("novel", "new", "first", "unknown")
                    ),
                },
                "citation_components": {
                    "entities_in_hypothesis": self._extract_entities_from_desc(desc),
                    "top_mention_count": max(
                        (
                            entity_mentions.get(ent, {}).get("mention_count", 0)
                            for ent in self._extract_entities_from_desc(desc)
                        ),
                        default=0,
                    ),
                },
                "relation_components": {
                    "high_impact_relations_found": relation_sig > 0.5,
                    "relation_signal_contribution": relation_sig,
                },
            }

            scores.append(
                HypothesisImpactScore(
                    hypothesis_id=hyp_id,
                    hypothesis_type=hyp_type,
                    impact_score=impact,
                    novelty_signal=novelty_sig,
                    citation_signal=citation_sig,
                    relation_signal=relation_sig,
                    base_confidence=base_conf,
                    total_signal=total_sig,
                    provenance=provenance,
                )
            )

        return scores

    def _compute_novelty_signal(
        self,
        hyp_type: str,
        description: str,
        entity_mentions: dict[str, Any],
    ) -> float:
        """Compute novelty signal: gaps and contradictions are novel."""
        base = 0.5

        # Gap and contradiction hypotheses are inherently novel
        if hyp_type == "gap":
            base += 0.25
        if hyp_type == "contradiction":
            base += 0.3

        # Description keywords signaling novelty
        novel_keywords = ("novel", "new", "first", "unknown", "missing", "unconnected")
        if any(kw in description.lower() for kw in novel_keywords):
            base += 0.15

        # Novelty boost if hypothesis involves rarely-mentioned entities
        hypothesis_entities = self._extract_entities_from_desc(description)
        if hypothesis_entities:
            mention_counts = [
                entity_mentions.get(ent, {}).get("mention_count", 0) for ent in hypothesis_entities
            ]
            if mention_counts and sum(mention_counts) < len(mention_counts) * 2:
                # Entities mentioned only 1-2 times → rare → novel
                base += 0.1

        return min(1.0, base)

    def _compute_citation_signal(
        self,
        description: str,
        entity_mentions: dict[str, Any],
    ) -> float:
        """Compute citation signal: importance via mention frequency."""
        base = 0.4

        # Extract entities from description
        hypothesis_entities = self._extract_entities_from_desc(description)

        if not hypothesis_entities:
            return base

        mention_counts = [
            entity_mentions.get(ent, {}).get("mention_count", 0) for ent in hypothesis_entities
        ]

        if not mention_counts:
            return base

        # Normalize mention frequency
        avg_mentions = sum(mention_counts) / len(mention_counts)
        max_mentions = max(mention_counts)

        # Logarithmic boost: many mentions → important field
        import math

        mention_boost = min(0.4, math.log(max_mentions + 1) * 0.1)
        base += mention_boost

        return min(1.0, base)

    def _compute_relation_signal(
        self,
        description: str,
        relations: list[dict[str, Any]],
    ) -> float:
        """Compute relation signal: hypothesis involves high-impact research relations."""
        base = 0.4

        if not relations:
            return base

        # Extract entities from hypothesis description
        hypothesis_entities = self._extract_entities_from_desc(description)

        if not hypothesis_entities:
            return base

        # Find relations involving these entities
        matching_relations = []
        for rel in relations:
            if not isinstance(rel, dict):
                continue
            subject = rel.get("subject_span", "")
            obj = rel.get("object_span", "")
            for ent in hypothesis_entities:
                if ent.lower() in subject.lower() or ent.lower() in obj.lower():
                    matching_relations.append(rel)
                    break

        if not matching_relations:
            return base

        # Weight by relation type impact
        max_weight = 0.0
        for rel in matching_relations:
            rel_type = rel.get("relation_type", "RELATED_TO")
            rel_conf = rel.get("confidence", 0.5)
            type_weight = self.RELATION_TYPE_WEIGHTS.get(rel_type, 0.5)
            weighted = type_weight * rel_conf
            max_weight = max(max_weight, weighted)

        base += min(0.4, max_weight * 0.5)

        return min(1.0, base)

    def _extract_entities_from_desc(self, description: str) -> list[str]:
        """Extract quoted entity names from description."""
        import re

        # Match quoted strings like 'BERT' or "temperature"
        matches = re.findall(r"['\"]([^'\"]+)['\"]", description)
        return [m.strip() for m in matches if m.strip()]
