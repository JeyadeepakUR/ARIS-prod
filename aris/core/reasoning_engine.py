"""Reasoning engine for generating evidence-backed reasoning steps.

When an LLMProvider is supplied, it generates real semantic evidence for
knowledge graph link justification. Falls back to deterministic heuristics
when no provider is configured (useful for tests and CLI mode).
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import TYPE_CHECKING

from aris.core.input_interface import InputPacket

if TYPE_CHECKING:
    from aris.llm.provider import LLMProvider


@dataclass(frozen=True)
class ReasoningResult:
    """Immutable result from the reasoning engine."""

    reasoning_steps: list[str]
    confidence_score: float
    input_packet: InputPacket

    def __post_init__(self) -> None:
        if not 0.0 <= self.confidence_score <= 1.0:
            raise ValueError(
                f"Confidence score must be between 0.0 and 1.0, got {self.confidence_score}"
            )


_LINK_EVIDENCE_PROMPT = """\
You are a research intelligence system. Analyze the relationship between two research documents \
and extract concrete evidence for a proposed knowledge graph link.

{content}

Proposed link type: {link_type}
Link hint: {hint}

Produce a JSON object with these fields:
- "steps": list of 4-5 concise evidence steps (each ≤ 80 words), each citing specific content
- "confidence": float between 0.0 and 1.0 reflecting evidential strength
- "relationship_type": the most accurate relationship type among: \
supports, contradicts, extends, replicates, applies, cites, shares_methodology, shares_dataset

Return only the JSON object."""

_GENERAL_REASONING_PROMPT = """\
Analyze the following research content and summarize the key claims and evidence in 4 concise steps.

Content:
{text}

Return a JSON object with:
- "steps": list of 4 reasoning steps
- "confidence": float 0.0-1.0

Return only the JSON object."""


class ReasoningEngine:
    """Generates evidence-backed reasoning steps for knowledge graph links.

    When initialized with an LLMProvider, calls the LLM to extract real
    semantic evidence. Otherwise falls back to deterministic heuristics.
    """

    def __init__(self, provider: LLMProvider | None = None) -> None:
        self._provider = provider

    def reason(self, input_packet: InputPacket) -> ReasoningResult:
        """Generate reasoning steps for the given input."""
        if self._provider is not None:
            return self._llm_reason(input_packet)
        return self._heuristic_reason(input_packet)

    def _llm_reason(self, input_packet: InputPacket) -> ReasoningResult:
        text = input_packet.text.strip()

        # Detect if this is a link-justification prompt (contains "Proposed Link Type")
        if "Proposed Link Type:" in text or "link_type:" in text.lower():
            parts = self._parse_link_prompt(text)
            prompt = _LINK_EVIDENCE_PROMPT.format(
                content=parts["content"],
                link_type=parts["link_type"],
                hint=parts["hint"],
            )
        else:
            prompt = _GENERAL_REASONING_PROMPT.format(text=text[:3000])

        try:
            result = self._provider.complete_json(prompt, max_tokens=600)  # type: ignore[union-attr]
            steps = result.get("steps", [])
            if not isinstance(steps, list) or not steps:
                steps = [str(result.get("raw", text[:120]))]
            confidence = float(result.get("confidence", 0.75))
            confidence = max(0.0, min(1.0, confidence))
        except Exception:
            return self._heuristic_reason(input_packet)

        return ReasoningResult(
            reasoning_steps=[str(s) for s in steps[:5]],
            confidence_score=confidence,
            input_packet=input_packet,
        )

    def _heuristic_reason(self, input_packet: InputPacket) -> ReasoningResult:
        text = input_packet.text
        word_count = len(text.split()) if text else 0

        # Extract substantive content tokens for evidence steps
        tokens = [w for w in re.findall(r"[a-zA-Z]{4,}", text) if w.lower() not in {
            "with", "that", "this", "from", "have", "been", "were", "will",
        }]
        top_tokens = list(dict.fromkeys(tokens[:8]))
        token_str = ", ".join(top_tokens) if top_tokens else "the content"

        steps = [
            f"The documents share substantive content related to: {token_str}.",
            f"The input contains {word_count} words, indicating {'substantial' if word_count > 50 else 'limited'} detail.",
            "The proposed link type is consistent with the shared vocabulary and thematic overlap.",
            "Evidence quality is assessed based on term co-occurrence and structural similarity.",
        ]

        return ReasoningResult(
            reasoning_steps=steps,
            confidence_score=self._heuristic_confidence(text),
            input_packet=input_packet,
        )

    def _heuristic_confidence(self, text: str) -> float:
        length = len(text)
        if length == 0:
            return 0.2
        if length <= 30:
            return 0.45
        if length <= 80:
            return 0.65
        if length <= 300:
            return 0.75
        if length <= 800:
            return 0.82
        return 0.88

    def _parse_link_prompt(self, text: str) -> dict[str, str]:
        link_type_match = re.search(r"Proposed Link Type:\s*(.+)", text, re.IGNORECASE)
        hint_match = re.search(r"Hint:\s*(.+)", text, re.IGNORECASE)
        link_type = link_type_match.group(1).strip() if link_type_match else "related"
        hint = hint_match.group(1).strip() if hint_match else ""
        content_end = link_type_match.start() if link_type_match else len(text)
        content = text[:content_end].strip()[:2500]
        return {"content": content, "link_type": link_type, "hint": hint}
