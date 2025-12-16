"""Tests for deterministic evaluation heuristics (Module 3)."""

from __future__ import annotations

import uuid
from datetime import UTC, datetime

from aris.core.evaluation import Evaluator
from aris.core.input_interface import InputPacket
from aris.core.reasoning_engine import ReasoningEngine, ReasoningResult


def _make_packet(text: str) -> InputPacket:
    return InputPacket(
        text=text,
        source="unit-test",
        request_id=uuid.uuid4(),
        timestamp=datetime.now(UTC),
    )


def test_evaluation_is_deterministic() -> None:
    engine = ReasoningEngine()
    evaluator = Evaluator()
    packet = _make_packet("consistent input")

    result = engine.reason(packet)
    eval1 = evaluator.evaluate(result)
    eval2 = evaluator.evaluate(result)

    assert eval1 == eval2


def test_metrics_within_bounds() -> None:
    engine = ReasoningEngine()
    evaluator = Evaluator()
    packet = _make_packet("bounded metrics")

    result = engine.reason(packet)
    evaluation = evaluator.evaluate(result)

    for value in (
        evaluation.step_count_score,
        evaluation.coherence_score,
        evaluation.confidence_alignment_score,
        evaluation.input_coverage_score,
        evaluation.overall_score,
    ):
        assert 0.0 <= value <= 1.0


def test_pathological_step_count_penalized() -> None:
    evaluator = Evaluator()
    packet = _make_packet("short")
    many_steps = [f"Step {i}" for i in range(12)]
    result = ReasoningResult(
        reasoning_steps=many_steps,
        confidence_score=0.5,
        input_packet=packet,
    )

    evaluation = evaluator.evaluate(result)

    assert evaluation.step_count_score == 0.0
    assert evaluation.overall_score < 1.0


def test_overconfident_answer_penalized() -> None:
    evaluator = Evaluator()
    packet = _make_packet("tiny")
    result = ReasoningResult(
        reasoning_steps=["only step"],
        confidence_score=1.0,
        input_packet=packet,
    )

    evaluation = evaluator.evaluate(result)

    assert evaluation.confidence_alignment_score < 0.5


def test_input_coverage_full_match() -> None:
    evaluator = Evaluator()
    packet = _make_packet("alpha beta")
    steps = ["Step 1 alpha", "Step 2 beta"]
    result = ReasoningResult(
        reasoning_steps=steps,
        confidence_score=0.6,
        input_packet=packet,
    )

    evaluation = evaluator.evaluate(result)

    assert evaluation.input_coverage_score == 1.0


def test_evaluation_does_not_mutate_inputs() -> None:
    evaluator = Evaluator()
    packet = _make_packet("immutable test")
    steps = ["s1", "s2"]
    result = ReasoningResult(
        reasoning_steps=list(steps),
        confidence_score=0.7,
        input_packet=packet,
    )

    evaluator.evaluate(result)

    assert result.reasoning_steps == steps
