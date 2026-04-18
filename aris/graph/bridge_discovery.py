"""Cross-domain bridge concept discovery for innovation mining."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from aris.core.semantic_analyzer import SemanticProfile

if TYPE_CHECKING:
    from aris.llm.provider import LLMProvider


@dataclass(frozen=True)
class BridgeCandidate:
    """Potential cross-domain bridge concept with novelty signal."""

    source_domain: str
    target_domain: str
    bridge_concept: str
    shared_keywords: tuple[str, ...]
    novelty: float
    rationale: str


_BRIDGE_RATIONALE_PROMPT = """\
Two research domains share a common concept that may enable cross-domain knowledge transfer.

Source domain: {source_domain}
Target domain: {target_domain}
Bridge concept: {bridge_concept}
Shared keywords: {shared_keywords}

Explain in 1-2 sentences why this concept bridges these domains and what specific \
transfer mechanism it enables. Be precise and research-oriented.

Return only the rationale text (no JSON, no preamble)."""


class BridgeDiscoveryEngine:
    """Detect cross-domain bridge concepts from semantic profiles.

    When an LLMProvider is supplied, generates specific rationale text explaining
    the transfer mechanism. Falls back to generic rationale otherwise.
    """

    def __init__(self, provider: LLMProvider | None = None) -> None:
        self._provider = provider

    def discover(self, profiles: list[SemanticProfile], *, top_k: int = 25) -> list[BridgeCandidate]:
        grouped = self._group_by_domain(profiles)
        domains = sorted(grouped.keys())
        bridges: list[BridgeCandidate] = []

        for idx, source in enumerate(domains):
            for target in domains[idx + 1:]:
                bridges.extend(self._domain_pair(source, target, grouped[source], grouped[target]))

        bridges_sorted = sorted(
            bridges,
            key=lambda bridge: (-bridge.novelty, bridge.source_domain, bridge.target_domain),
        )
        top = bridges_sorted[: max(1, top_k)]

        # Enrich top bridges with LLM rationale when provider is available
        if self._provider is not None:
            top = [self._enrich_rationale(b) for b in top]

        return top

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

        # Include n-gram concepts from profiles
        source_ngrams = self._ngram_strength(source_profiles)
        target_ngrams = self._ngram_strength(target_profiles)
        source_keywords.update(source_ngrams)
        target_keywords.update(target_ngrams)

        shared = sorted(set(source_keywords).intersection(target_keywords))
        if not shared:
            return []

        candidates: list[BridgeCandidate] = []
        for concept in shared:
            source_strength = source_keywords[concept]
            target_strength = target_keywords[concept]

            # Improved novelty: asymmetry + absolute prominence
            asymmetry = abs(source_strength - target_strength) / max(source_strength + target_strength, 1e-9)
            prominence = min(1.0, (source_strength + target_strength) * 10)
            novelty = round(min(1.0, 0.1 + 0.6 * asymmetry + 0.3 * prominence), 3)

            if novelty < 0.08:
                continue

            shared_keywords_list = [
                k for k in shared if k != concept
            ][:7]

            rationale = (
                f"Concept '{concept}' appears prominently in both {source_domain} and {target_domain} "
                f"with asymmetric usage patterns (novelty={novelty:.2f}), "
                "suggesting underexplored transfer opportunities."
            )
            candidates.append(
                BridgeCandidate(
                    source_domain=source_domain,
                    target_domain=target_domain,
                    bridge_concept=concept,
                    shared_keywords=tuple(shared_keywords_list),
                    novelty=novelty,
                    rationale=rationale,
                )
            )

        return candidates

    def _enrich_rationale(self, bridge: BridgeCandidate) -> BridgeCandidate:
        """Replace generic rationale with LLM-generated specific explanation."""
        try:
            prompt = _BRIDGE_RATIONALE_PROMPT.format(
                source_domain=bridge.source_domain,
                target_domain=bridge.target_domain,
                bridge_concept=bridge.bridge_concept,
                shared_keywords=", ".join(bridge.shared_keywords[:6]),
            )
            rationale = self._provider.complete(prompt, max_tokens=150, temperature=0.3)  # type: ignore[union-attr]
            rationale = rationale.strip()
            if rationale and len(rationale) > 20:
                return BridgeCandidate(
                    source_domain=bridge.source_domain,
                    target_domain=bridge.target_domain,
                    bridge_concept=bridge.bridge_concept,
                    shared_keywords=bridge.shared_keywords,
                    novelty=bridge.novelty,
                    rationale=rationale,
                )
        except Exception:
            pass
        return bridge

    def _keyword_strength(self, profiles: list[SemanticProfile]) -> dict[str, float]:
        strength: dict[str, float] = {}
        for profile in profiles:
            for signal in profile.keywords:
                strength[signal.token] = round(strength.get(signal.token, 0.0) + signal.weight, 6)
        return strength

    def _ngram_strength(self, profiles: list[SemanticProfile]) -> dict[str, float]:
        """Incorporate multi-word n-gram concepts for richer bridge detection."""
        strength: dict[str, float] = {}
        for profile in profiles:
            ngrams = getattr(profile, "ngram_concepts", ())
            for ngram in ngrams:
                strength[ngram] = round(strength.get(ngram, 0.0) + 0.01, 6)
        return strength
