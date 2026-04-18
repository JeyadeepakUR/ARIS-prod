"""Deterministic semantic analysis for multi-paper research intelligence.

Extracts normalized keywords (TF-IDF weighted), n-gram concepts, claim units,
section structure, citation signals, and domain-aware semantic profiles from
raw document text without ML dependencies.
"""

from __future__ import annotations

import math
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
class CitationSignal:
    """Citation reference extracted from a document."""

    raw: str
    style: str  # "numeric", "author_year", "unknown"
    position: int  # character offset in text


@dataclass(frozen=True)
class DocumentSection:
    """Detected document section with its text content."""

    name: str  # "abstract", "introduction", "methods", "results", "discussion", "conclusion", "body"
    text: str
    start: int
    end: int


@dataclass(frozen=True)
class SemanticProfile:
    """Document-level semantic profile for downstream engines."""

    document_id: str
    domain: str
    keywords: tuple[KeywordSignal, ...]
    claims: tuple[ClaimUnit, ...]
    concepts: tuple[str, ...]
    ngram_concepts: tuple[str, ...]
    sections: tuple[DocumentSection, ...]
    citations: tuple[CitationSignal, ...]
    research_entities: tuple[str, ...]


class SemanticAnalyzer:
    """Extract deterministic semantic profiles from research documents.

    Key capabilities:
    - TF-IDF weighting across a document corpus (pass corpus_texts for IDF)
    - Bigram/trigram multi-word concept extraction
    - Section boundary detection (Abstract, Introduction, Methods, Results, Discussion)
    - Citation pattern extraction ([1], (Author, 2023), etc.)
    - Research-specific named entity patterns (metrics, model names, dataset names)
    """

    _STOP_WORDS: frozenset[str] = frozenset({
        "a", "an", "and", "are", "as", "at", "be", "by", "do", "for",
        "from", "has", "have", "in", "is", "it", "its", "of", "on", "or",
        "that", "the", "to", "was", "were", "with", "we", "our", "this",
        "these", "those", "their", "they", "into", "than", "then", "also",
        "but", "not", "so", "if", "can", "may", "will", "would", "could",
        "should", "been", "being", "had", "have", "which", "who", "when",
        "where", "how", "what", "while", "thus", "hence", "both", "each",
        "such", "more", "most", "some", "any", "all", "no", "nor", "yet",
        "use", "used", "using", "show", "shows", "shown", "since", "per",
    })

    _CLAIM_MARKERS: tuple[str, ...] = (
        "improves", "outperforms", "achieves", "reduces", "increases",
        "fails", "degrades", "contradicts", "supports", "does not",
        "cannot", "better", "worse", "significantly", "substantially",
        "surpasses", "exceeds", "demonstrates", "shows that", "we show",
        "we find", "results show", "experiments show", "we propose",
        "we introduce", "our method", "our approach", "state-of-the-art",
        "state of the art", "novel", "outperform",
    )

    _POSITIVE_MARKERS: tuple[str, ...] = (
        "improves", "outperforms", "achieves", "supports", "better",
        "increases", "surpasses", "exceeds", "demonstrates", "state-of-the-art",
        "significantly", "substantially", "novel",
    )

    _NEGATIVE_MARKERS: tuple[str, ...] = (
        "fails", "degrades", "contradicts", "worse", "does not",
        "cannot", "reduces", "limitation", "insufficient",
    )

    # Research metrics
    _METRIC_NAMES: frozenset[str] = frozenset({
        "accuracy", "f1", "precision", "recall", "bleu", "rouge", "perplexity",
        "auc", "map", "ndcg", "mrr", "psnr", "ssim", "fid", "mse", "mae",
        "rmse", "r2", "iou", "dice", "cer", "wer", "top-1", "top-5",
        "throughput", "latency", "flops", "parameters",
    })

    # Known model/method names (lowercase)
    _MODEL_NAMES: frozenset[str] = frozenset({
        "bert", "gpt", "gpt-2", "gpt-3", "gpt-4", "roberta", "xlnet", "albert",
        "t5", "bart", "llama", "mistral", "falcon", "gemini", "claude",
        "resnet", "vgg", "inception", "efficientnet", "vit", "deit", "clip",
        "dalle", "stable diffusion", "diffusion", "transformer", "lstm", "gru",
        "cnn", "rnn", "gan", "vae", "ddpm", "sam", "yolo", "detr",
    })

    # Section header patterns
    _SECTION_PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = (
        ("abstract", re.compile(r"^\s*abstract\s*$", re.IGNORECASE | re.MULTILINE)),
        ("introduction", re.compile(r"^\s*(?:\d+\.?\s*)?introduction\s*$", re.IGNORECASE | re.MULTILINE)),
        ("methods", re.compile(
            r"^\s*(?:\d+\.?\s*)?(?:methods?|methodology|approach|proposed method|materials and methods)\s*$",
            re.IGNORECASE | re.MULTILINE,
        )),
        ("results", re.compile(r"^\s*(?:\d+\.?\s*)?(?:results?|experiments?|evaluation)\s*$", re.IGNORECASE | re.MULTILINE)),
        ("discussion", re.compile(r"^\s*(?:\d+\.?\s*)?discussion\s*$", re.IGNORECASE | re.MULTILINE)),
        ("conclusion", re.compile(
            r"^\s*(?:\d+\.?\s*)?(?:conclusions?|concluding remarks?|summary)\s*$",
            re.IGNORECASE | re.MULTILINE,
        )),
        ("related_work", re.compile(
            r"^\s*(?:\d+\.?\s*)?(?:related work|background|prior work|literature review)\s*$",
            re.IGNORECASE | re.MULTILINE,
        )),
    )

    _NUMERIC_CITATION: re.Pattern[str] = re.compile(r"\[(\d+(?:,\s*\d+)*)\]")
    _AUTHOR_YEAR_CITATION: re.Pattern[str] = re.compile(
        r"\(([A-Z][a-zA-Z]+(?:\s+(?:et al\.?|and\s+[A-Z][a-zA-Z]+))?,\s*(?:19|20)\d{2}[a-z]?)\)"
    )

    def analyze(
        self,
        text: str,
        *,
        document_id: str,
        domain: str,
        corpus_texts: list[str] | None = None,
    ) -> SemanticProfile:
        """Build semantic profile from input text deterministically.

        Args:
            text: Raw document text.
            document_id: Unique document identifier.
            domain: Document domain label.
            corpus_texts: Other document texts for IDF computation.
        """
        normalized = self._normalize_text(text)
        sections = self._detect_sections(text)
        keywords = self._extract_keywords(normalized, corpus_texts=corpus_texts)
        claims = self._extract_claims(normalized)
        concepts = self._extract_concepts(text)
        ngrams = self._extract_ngrams(normalized)
        citations = self._extract_citations(text)
        entities = self._extract_research_entities(text)

        return SemanticProfile(
            document_id=document_id,
            domain=domain.strip().lower() or "unknown",
            keywords=tuple(keywords),
            claims=tuple(claims),
            concepts=tuple(concepts),
            ngram_concepts=tuple(ngrams),
            sections=tuple(sections),
            citations=tuple(citations),
            research_entities=tuple(entities),
        )

    def _normalize_text(self, text: str) -> str:
        collapsed = re.sub(r"[ \t\f\v]+", " ", text.replace("\r\n", "\n").replace("\r", "\n"))
        collapsed = re.sub(r"\n{3,}", "\n\n", collapsed)
        return collapsed.strip()

    def _extract_keywords(
        self,
        text: str,
        *,
        corpus_texts: list[str] | None,
    ) -> list[KeywordSignal]:
        tokens = self._tokenize(text)
        counts: dict[str, int] = {}
        for token in tokens:
            if token in self._STOP_WORDS or len(token) < 3:
                continue
            counts[token] = counts.get(token, 0) + 1

        total = max(1, sum(counts.values()))
        tf: dict[str, float] = {tok: cnt / total for tok, cnt in counts.items()}

        if corpus_texts:
            n_docs = len(corpus_texts) + 1
            idf: dict[str, float] = {}
            for tok in tf:
                df = sum(1 for doc in corpus_texts if tok in doc.lower()) + 1
                idf[tok] = math.log(n_docs / df)
            weights = {tok: tf[tok] * idf[tok] for tok in tf}
        else:
            weights = tf

        ranked = sorted(weights.items(), key=lambda item: (-item[1], item[0]))
        top = ranked[:25]

        return [
            KeywordSignal(token=tok, count=counts[tok], weight=round(weights[tok], 6))
            for tok, _ in top
        ]

    def _extract_claims(self, text: str) -> list[ClaimUnit]:
        sentences = self._split_sentences(text)
        claims: list[ClaimUnit] = []

        for sentence in sentences:
            lowered = sentence.lower()
            marker_hits = [m for m in self._CLAIM_MARKERS if m in lowered]
            if not marker_hits:
                continue

            polarity = self._polarity(lowered)
            confidence = min(1.0, 0.45 + 0.08 * len(marker_hits))

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
        normalized = sorted({c.strip() for c in candidates if c.lower() not in self._STOP_WORDS})
        return normalized[:40]

    def _extract_ngrams(self, text: str) -> list[str]:
        tokens = self._tokenize(text)
        filtered = [t for t in tokens if t not in self._STOP_WORDS and len(t) >= 3]
        bigrams: dict[str, int] = {}
        trigrams: dict[str, int] = {}

        for i in range(len(filtered) - 1):
            bg = f"{filtered[i]} {filtered[i + 1]}"
            bigrams[bg] = bigrams.get(bg, 0) + 1

        for i in range(len(filtered) - 2):
            tg = f"{filtered[i]} {filtered[i + 1]} {filtered[i + 2]}"
            trigrams[tg] = trigrams.get(tg, 0) + 1

        min_count = 2
        good_bg = [bg for bg, cnt in bigrams.items() if cnt >= min_count]
        good_tg = [tg for tg, cnt in trigrams.items() if cnt >= min_count]

        suppressed: set[str] = set()
        for tg in good_tg:
            parts = tg.split()
            suppressed.add(f"{parts[0]} {parts[1]}")
            suppressed.add(f"{parts[1]} {parts[2]}")

        result = sorted(set(good_tg) | {bg for bg in good_bg if bg not in suppressed})
        return result[:30]

    def _detect_sections(self, text: str) -> list[DocumentSection]:
        hits: list[tuple[str, int]] = []

        for section_name, pattern in self._SECTION_PATTERNS:
            for match in pattern.finditer(text):
                hits.append((section_name, match.start()))

        if not hits:
            return [DocumentSection(name="body", text=text.strip(), start=0, end=len(text))]

        hits.sort(key=lambda h: h[1])
        sections: list[DocumentSection] = []
        for idx, (name, start) in enumerate(hits):
            end = hits[idx + 1][1] if idx + 1 < len(hits) else len(text)
            section_text = text[start:end].strip()
            sections.append(DocumentSection(name=name, text=section_text, start=start, end=end))

        return sections

    def _extract_citations(self, text: str) -> list[CitationSignal]:
        citations: list[CitationSignal] = []

        for match in self._NUMERIC_CITATION.finditer(text):
            citations.append(CitationSignal(raw=match.group(0), style="numeric", position=match.start()))

        for match in self._AUTHOR_YEAR_CITATION.finditer(text):
            citations.append(CitationSignal(raw=match.group(0), style="author_year", position=match.start()))

        citations.sort(key=lambda c: c.position)
        return citations[:100]

    def _extract_research_entities(self, text: str) -> list[str]:
        lowered = text.lower()
        found: set[str] = set()

        for metric in self._METRIC_NAMES:
            if metric in lowered:
                found.add(f"metric:{metric}")

        for model in self._MODEL_NAMES:
            if model in lowered:
                found.add(f"model:{model}")

        pct_matches = re.findall(r"\b(\d+(?:\.\d+)?)\s*%", text)
        if pct_matches:
            found.add(f"quantitative_results:{len(pct_matches)}_percentages")

        dataset_pattern = re.compile(
            r"\b([A-Z][A-Za-z0-9\-]+(?:\s+[A-Z][A-Za-z0-9\-]+)?)\s+"
            r"(?:dataset|benchmark|corpus|database|collection)\b",
            re.IGNORECASE,
        )
        for match in dataset_pattern.finditer(text):
            found.add(f"dataset:{match.group(1).strip()}")

        return sorted(found)[:50]

    def _tokenize(self, text: str) -> list[str]:
        cleaned = re.sub(r"[^a-zA-Z0-9\s\-]", " ", text.lower())
        return [token for token in cleaned.split() if token and len(token) >= 2]

    def _split_sentences(self, text: str) -> list[str]:
        parts = re.split(r"(?<=[.!?])\s+", text)
        return [part.strip() for part in parts if len(part.strip()) >= 20]

    def _polarity(self, lowered_sentence: str) -> str:
        pos_hits = sum(1 for marker in self._POSITIVE_MARKERS if marker in lowered_sentence)
        neg_hits = sum(1 for marker in self._NEGATIVE_MARKERS if marker in lowered_sentence)
        if pos_hits > neg_hits:
            return "positive"
        if neg_hits > pos_hits:
            return "negative"
        return "neutral"
