"""Tests for SciBERT Role Induction Tool (Module 10)."""

from __future__ import annotations

import json
from typing import Callable

import pytest

from aris.ml.role_induction import RoleCandidate, SciBertRoleTool


def _make_stub_candidate(text: str, start: int, end: int, role: str, confidence: float) -> RoleCandidate:
    return RoleCandidate(
        span_text=text[start:end],
        start=start,
        end=end,
        role=role,
        confidence=confidence,
        provenance={"model": "stub", "tokenizer": "stub", "role_vocab": [role], "strategy": "stub"},
    )


def _patch_infer(tool: SciBertRoleTool, factory: Callable[[str], list[RoleCandidate]]) -> None:
    tool._infer_roles = factory  # type: ignore[attr-defined]


def test_determinism(monkeypatch: pytest.MonkeyPatch) -> None:
    tool = SciBertRoleTool()

    def deterministic_factory(text: str) -> list[RoleCandidate]:
        return [
            _make_stub_candidate(text, 0, min(5, len(text)), "background", 0.62),
            _make_stub_candidate(text, max(0, len(text) - 4), len(text), "result", 0.58),
        ]

    _patch_infer(tool, deterministic_factory)

    first = json.loads(tool.execute("abcde"))
    second = json.loads(tool.execute("abcde"))

    assert first == second
    assert all(isinstance(item["confidence"], float) for item in first)


def test_span_alignment(monkeypatch: pytest.MonkeyPatch) -> None:
    text = "Alice studies biology."
    tool = SciBertRoleTool()

    def span_factory(src: str) -> list[RoleCandidate]:
        start = src.index("Alice")
        end = start + len("Alice")
        return [_make_stub_candidate(src, start, end, "background", 0.71)]

    _patch_infer(tool, span_factory)

    output = json.loads(tool.execute(text))
    assert len(output) == 1
    candidate = output[0]
    assert candidate["start"] == 0
    assert candidate["end"] == len("Alice")
    assert candidate["span_text"] == "Alice"


def test_confidence_bounds() -> None:
    text = "Short note."
    tool = SciBertRoleTool()

    def bounds_factory(src: str) -> list[RoleCandidate]:
        return [
            _make_stub_candidate(src, 0, len(src), "finding", 1.5),
            _make_stub_candidate(src, 0, 5, "method", -0.2),
        ]

    _patch_infer(tool, bounds_factory)

    output = json.loads(tool.execute(text))
    assert all(0.0 <= item["confidence"] <= 1.0 for item in output)


def test_failure_isolation(monkeypatch: pytest.MonkeyPatch) -> None:
    tool = SciBertRoleTool()

    def failing_factory(_: str) -> list[RoleCandidate]:
        raise RuntimeError("inference failed")

    _patch_infer(tool, failing_factory)

    output = json.loads(tool.execute("anything"))
    assert "error" in output
    assert output.get("error_type") == "RuntimeError"
    assert output.get("model") == tool.MODEL_NAME
