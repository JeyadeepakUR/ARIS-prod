"""Deterministic semantic analysis for multi-paper research intelligence.

This module extracts normalized keywords, claim units, and domain-aware
semantic profiles from raw document text without ML dependencies.
"""

from __future__ import annotations

import re
from dataclasses import dataclass


@dataclass(frozen=True)
class KeywordSignal:
    """Weighted keyword extracted from a document."""

    token: str
    count: int
    weight: float


@dataclass(frozen=True)
class ClaimUnit:
    """Single normalized claim extracted from document text."""

    text: str
    polarity: str
    confidence: float
    evidence_sentence: str


@dataclass(frozen=True)
class SemanticProfile:
    """Document-level semantic profile for downstream engines."""

    document_id: str
    domain: str
    keywords: tuple[KeywordSignal, ...]
    claims: tuple[ClaimUnit, ...]
    concepts: tuple[str, ...]


class SemanticAnalyzer:
    """Extract deterministic semantic profiles from research documents."""

    _STOP_WORDS: set[str] = {
        "a",
        "an",
        "and",
        "are",
        "as",
        "at",
        "be",
        "by",
        "for",
        "from",
        "has",
        "in",
        "is",
        "it",
        "its",
        "of",
        "on",
        "or",
        "that",
        "the",
        "to",
        "was",
        "were",
        "with",
        "we",
        "our",
        "this",
        "these",
        "those",
        "their",
        "they",
        "into",
        "than",
        "then",
    }

    _CLAIM_MARKERS: tuple[str, ...] = (
        "improves",
        "outperforms",
        "achieves",
        "reduces",
        "increases",
        "fails",
        "degrades",
        "contradicts",
        "supports",
        "does not",
        "cannot",
        "better",
        "worse",
    )

    _POSITIVE_MARKERS: tuple[str, ...] = (
        "improves",
        "outperforms",
        "achieves",
        "supports",
        "better",
        "increases",
    )

    _NEGATIVE_MARKERS: tuple[str, ...] = (
        "fails",
        "degrades",
        "contradicts",
        "worse",
        "does not",
        "cannot",
        "reduces",
    )

    def analyze(self, text: str, *, document_id: str, domain: str) -> SemanticProfile:
        """Build semantic profile from input text deterministically."""

        normalized = self._normalize_text(text)
        keywords = self._extract_keywords(normalized)
        claims = self._extract_claims(normalized)
        concepts = self._extract_concepts(text)

        return SemanticProfile(
            document_id=document_id,
            domain=domain.strip().lower() or "unknown",
            keywords=tuple(keywords),
            claims=tuple(claims),
            concepts=tuple(concepts),
        )

    def _normalize_text(self, text: str) -> str:
        collapsed = re.sub(r"\s+", " ", text.replace("\n", " ").strip())
        return collapsed

    def _extract_keywords(self, text: str) -> list[KeywordSignal]:
        tokens = self._tokenize(text)
        counts: dict[str, int] = {}
        for token in tokens:
            if token in self._STOP_WORDS:
                continue
            counts[token] = counts.get(token, 0) + 1

        total = max(1, sum(counts.values()))
        ranked = sorted(counts.items(), key=lambda item: (-item[1], item[0]))
        top = ranked[:20]

        return [
            KeywordSignal(token=token, count=count, weight=round(count / total, 6))
            for token, count in top
        ]

    def _extract_claims(self, text: str) -> list[ClaimUnit]:
        sentences = self._split_sentences(text)
        claims: list[ClaimUnit] = []

        for sentence in sentences:
            lowered = sentence.lower()
            if not any(marker in lowered for marker in self._CLAIM_MARKERS):
                continue

            polarity = self._polarity(lowered)
            marker_hits = sum(1 for m in self._CLAIM_MARKERS if m in lowered)
            confidence = min(1.0, 0.45 + 0.1 * marker_hits)

            claims.append(
                ClaimUnit(
                    text=sentence,
                    polarity=polarity,
                    confidence=round(confidence, 3),
                    evidence_sentence=sentence,
                )
            )

        return claims

    def _extract_concepts(self, raw_text: str) -> list[str]:
        candidates = re.findall(r"\b[A-Z][A-Za-z0-9\-]{2,}\b", raw_text)
        normalized = sorted({candidate.strip() for candidate in candidates})
        return normalized[:30]

    def _tokenize(self, text: str) -> list[str]:
        cleaned = re.sub(r"[^a-zA-Z0-9\s]", " ", text.lower())
        return [token for token in cleaned.split() if token]

    def _split_sentences(self, text: str) -> list[str]:
        parts = re.split(r"(?<=[.!?])\s+", text)
        return [part.strip() for part in parts if part.strip()]

    def _polarity(self, lowered_sentence: str) -> str:
        pos_hits = sum(1 for marker in self._POSITIVE_MARKERS if marker in lowered_sentence)
        neg_hits = sum(1 for marker in self._NEGATIVE_MARKERS if marker in lowered_sentence)
        if pos_hits > neg_hits:
            return "positive"
        if neg_hits > pos_hits:
            return "negative"
        return "neutral"
