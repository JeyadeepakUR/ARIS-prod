"""Governance tests for optional ML boundary (Module 11).

These tests do not require ML deps; they simulate failure modes and
verify structured error payloads and safe imports.
"""

from __future__ import annotations

import json

import pytest


def test_phase1_imports_are_independent() -> None:
    # Import Phase 1 graph package without pulling ML
    import aris.graph as _  # noqa: F401


def test_ml_module_import_is_safe_without_deps() -> None:
    # Ontology module import should be safe (lazy imports inside functions)
    import aris.ml.ontology_induction as _  # noqa: F401


def test_tool_returns_structured_error_when_hdbscan_missing(monkeypatch: pytest.MonkeyPatch) -> None:
    from aris.ml.ontology_induction import OntologyClusteringTool, _MissingOptionalDependency

    tool = OntologyClusteringTool()

    def raise_missing(_: list[list[float]]):  # type: ignore[no-untyped-def]
        raise _MissingOptionalDependency({"hdbscan", "numpy"})

    monkeypatch.setattr(tool, "_hdbscan_cluster", raise_missing)

    items = [{"id": "a", "text": "x"}, {"id": "b", "text": "y"}]
    payload = json.loads(tool.execute(json.dumps(items)))

    assert payload["error"] == "Missing optional dependency"
    assert payload["module"] == "ontology_induction"
    assert sorted(payload["required"]) == ["hdbscan", "numpy"]
    assert "aris[ml]" in payload["install_hint"]


def test_tool_returns_structured_error_when_transformers_missing(monkeypatch: pytest.MonkeyPatch) -> None:
    from aris.ml.ontology_induction import OntologyClusteringTool, _MissingOptionalDependency

    tool = OntologyClusteringTool()

    def ensure_model_missing() -> None:  # type: ignore[no-untyped-def]
        raise _MissingOptionalDependency({"torch", "transformers"})

    monkeypatch.setattr(tool, "_ensure_model", ensure_model_missing)

    # Trigger path through _embed -> _ensure_model
    items = [{"id": "a", "text": "x"}]
    payload = json.loads(tool.execute(json.dumps(items)))

    assert payload["error"] == "Missing optional dependency"
    assert payload["module"] == "ontology_induction"
    assert sorted(payload["required"]) == ["torch", "transformers"]
    assert "aris[ml]" in payload["install_hint"]
