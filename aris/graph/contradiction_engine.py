"""Cross-paper contradiction detection for ARIS research intelligence."""

from __future__ import annotations

from dataclasses import dataclass

from aris.core.semantic_analyzer import ClaimUnit, SemanticProfile


@dataclass(frozen=True)
class ContradictionRecord:
    """Evidence-backed contradiction between two claim units."""

    document_a: str
    document_b: str
    claim_a: str
    claim_b: str
    overlap_terms: tuple[str, ...]
    severity: float
    rationale: str


class ContradictionEngine:
    """Detect pairwise contradictions across semantic profiles."""

    def find_contradictions(self, profiles: list[SemanticProfile]) -> list[ContradictionRecord]:
        results: list[ContradictionRecord] = []

        for idx, profile_a in enumerate(profiles):
            for profile_b in profiles[idx + 1 :]:
                results.extend(self._pairwise(profile_a, profile_b))

        return sorted(
            results,
            key=lambda rec: (-rec.severity, rec.document_a, rec.document_b, rec.claim_a),
        )

    def _pairwise(
        self,
        profile_a: SemanticProfile,
        profile_b: SemanticProfile,
    ) -> list[ContradictionRecord]:
        records: list[ContradictionRecord] = []

        for claim_a in profile_a.claims:
            terms_a = self._claim_terms(claim_a)
            if not terms_a:
                continue
            for claim_b in profile_b.claims:
                if claim_a.polarity == claim_b.polarity:
                    continue
                if claim_a.polarity == "neutral" or claim_b.polarity == "neutral":
                    continue

                terms_b = self._claim_terms(claim_b)
                overlap = tuple(sorted(terms_a.intersection(terms_b)))
                if len(overlap) < 2:
                    continue

                severity = self._severity(claim_a, claim_b, len(overlap))
                rationale = (
                    "Claims share topical terms but express opposing polarity "
                    "under potentially different contexts."
                )
                records.append(
                    ContradictionRecord(
                        document_a=profile_a.document_id,
                        document_b=profile_b.document_id,
                        claim_a=claim_a.text,
                        claim_b=claim_b.text,
                        overlap_terms=overlap,
                        severity=severity,
                        rationale=rationale,
                    )
                )

        return records

    def _claim_terms(self, claim: ClaimUnit) -> set[str]:
        tokens = [tok.lower() for tok in claim.text.split()]
        filtered = {tok.strip(".,:;!?()[]{}\"'") for tok in tokens if len(tok) > 3}
        return {tok for tok in filtered if tok}

    def _severity(self, claim_a: ClaimUnit, claim_b: ClaimUnit, overlap_count: int) -> float:
        base = (claim_a.confidence + claim_b.confidence) / 2.0
        overlap_boost = min(0.3, overlap_count * 0.05)
        return round(min(1.0, base + overlap_boost), 3)
