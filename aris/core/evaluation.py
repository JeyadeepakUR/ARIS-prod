"""Deterministic evaluation heuristics for ARIS reasoning outputs."""

from __future__ import annotations

from dataclasses import dataclass

from aris.core.reasoning_engine import ReasoningResult


@dataclass(frozen=True)
class EvaluationResult:
    """Immutable container for heuristic evaluation metrics."""

    step_count_score: float
    coherence_score: float
    confidence_alignment_score: float
    input_coverage_score: float
    overall_score: float


class Evaluator:
    """Compute deterministic heuristic scores for a reasoning result."""

    def evaluate(self, result: ReasoningResult) -> EvaluationResult:
        steps = result.reasoning_steps
        text = result.input_packet.text
        confidence = result.confidence_score

        step_score = self._step_count_score(steps)
        coherence_score = self._coherence_score(steps)
        confidence_score = self._confidence_alignment_score(text, confidence)
        coverage_score = self._input_coverage_score(text, steps)

        overall = (step_score + coherence_score + confidence_score + coverage_score) / 4.0

        return EvaluationResult(
            step_count_score=step_score,
            coherence_score=coherence_score,
            confidence_alignment_score=confidence_score,
            input_coverage_score=coverage_score,
            overall_score=overall,
        )

    def _step_count_score(self, steps: list[str]) -> float:
        target = 4.0
        count = float(len(steps))
        if count == 0:
            return 0.0
        score = 1.0 - abs(count - target) / target
        return max(0.0, min(1.0, score))

    def _coherence_score(self, steps: list[str]) -> float:
        if not steps:
            return 0.0
        non_empty = sum(1 for step in steps if step.strip())
        return non_empty / len(steps)

    def _confidence_alignment_score(self, text: str, confidence: float) -> float:
        expected = min(1.0, max(0.0, len(text) / 120.0))
        delta = abs(confidence - expected)
        return max(0.0, min(1.0, 1.0 - delta))

    def _input_coverage_score(self, text: str, steps: list[str]) -> float:
        words = self._tokenize(text)
        if not words:
            return 1.0
        step_tokens = set(self._tokenize("\n".join(steps)))
        covered = sum(1 for w in words if w in step_tokens)
        return covered / len(words)

    def _tokenize(self, text: str) -> list[str]:
        cleaned = [ch.lower() if ch.isalnum() else " " for ch in text]
        tokens = "".join(cleaned).split()
        return tokens
