"""Cross-domain bridge concept discovery for innovation mining."""

from __future__ import annotations

from dataclasses import dataclass

from aris.core.semantic_analyzer import SemanticProfile


@dataclass(frozen=True)
class BridgeCandidate:
    """Potential cross-domain bridge concept with novelty signal."""

    source_domain: str
    target_domain: str
    bridge_concept: str
    shared_keywords: tuple[str, ...]
    novelty: float
    rationale: str


class BridgeDiscoveryEngine:
    """Detect cross-domain bridge concepts from semantic profiles."""

    def discover(self, profiles: list[SemanticProfile], *, top_k: int = 25) -> list[BridgeCandidate]:
        grouped = self._group_by_domain(profiles)
        domains = sorted(grouped.keys())
        bridges: list[BridgeCandidate] = []

        for idx, source in enumerate(domains):
            for target in domains[idx + 1 :]:
                bridges.extend(self._domain_pair(source, target, grouped[source], grouped[target]))

        bridges_sorted = sorted(
            bridges,
            key=lambda bridge: (-bridge.novelty, bridge.source_domain, bridge.target_domain),
        )
        return bridges_sorted[: max(1, top_k)]

    def _group_by_domain(self, profiles: list[SemanticProfile]) -> dict[str, list[SemanticProfile]]:
        grouped: dict[str, list[SemanticProfile]] = {}
        for profile in profiles:
            grouped.setdefault(profile.domain, []).append(profile)
        return grouped

    def _domain_pair(
        self,
        source_domain: str,
        target_domain: str,
        source_profiles: list[SemanticProfile],
        target_profiles: list[SemanticProfile],
    ) -> list[BridgeCandidate]:
        source_keywords = self._keyword_strength(source_profiles)
        target_keywords = self._keyword_strength(target_profiles)

        shared = sorted(set(source_keywords).intersection(target_keywords))
        if not shared:
            return []

        candidates: list[BridgeCandidate] = []
        for concept in shared:
            source_strength = source_keywords[concept]
            target_strength = target_keywords[concept]
            novelty = round(min(1.0, 0.1 + abs(source_strength - target_strength)), 3)
            if novelty < 0.08:
                continue

            shared_keywords = tuple(shared[:8])
            rationale = (
                "Concept appears in both domains with asymmetric prominence; "
                "transfer experiments may reveal underexplored opportunities."
            )
            candidates.append(
                BridgeCandidate(
                    source_domain=source_domain,
                    target_domain=target_domain,
                    bridge_concept=concept,
                    shared_keywords=shared_keywords,
                    novelty=novelty,
                    rationale=rationale,
                )
            )

        return candidates

    def _keyword_strength(self, profiles: list[SemanticProfile]) -> dict[str, float]:
        strength: dict[str, float] = {}
        for profile in profiles:
            for signal in profile.keywords:
                strength[signal.token] = round(strength.get(signal.token, 0.0) + signal.weight, 6)
        return strength
