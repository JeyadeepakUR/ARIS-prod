"""
Unit tests for the deterministic run loop.
"""

from aris.input_interface import InputInterface
from aris.memory_store import InMemoryStore
from aris.reasoning_engine import ReasoningEngine
from aris.run_loop import run_loop


def test_run_loop_basic() -> None:
    input_interface = InputInterface()
    engine = ReasoningEngine()
    store = InMemoryStore()

    raw_inputs = ["Hello", "  world  ", ""]

    results = run_loop(raw_inputs, input_interface, engine, store)

    # One result per input
    assert len(results) == len(raw_inputs)

    # Memory store has same number of traces
    traces = store.list_all()
    assert len(traces) == len(raw_inputs)

    # Verify trimming and deterministic conclusion (last step present)
    assert results[0].input_packet.text == "Hello"
    assert results[1].input_packet.text == "world"
    assert results[2].input_packet.text == ""
    assert isinstance(results[0].reasoning_steps[-1], str)


def test_run_loop_determinism_same_inputs_same_conclusions() -> None:
    input_interface = InputInterface()
    engine = ReasoningEngine()

    inputs = [
        "Short",
        "This is a longer input",
        "",
    ]

    store1 = InMemoryStore()
    store2 = InMemoryStore()

    results1 = run_loop(inputs, input_interface, engine, store1)
    results2 = run_loop(inputs, input_interface, engine, store2)

    # Confidence scores and last steps depend only on input length/words
    conclusions1 = [r.reasoning_steps[-1] for r in results1]
    conclusions2 = [r.reasoning_steps[-1] for r in results2]
    confidences1 = [r.confidence_score for r in results1]
    confidences2 = [r.confidence_score for r in results2]

    assert conclusions1 == conclusions2
    assert confidences1 == confidences2


def test_run_loop_stores_full_traces() -> None:
    input_interface = InputInterface()
    engine = ReasoningEngine()
    store = InMemoryStore()

    raw_inputs = ["Trace test"]
    run_loop(raw_inputs, input_interface, engine, store)

    traces = store.list_all()
    assert len(traces) == 1
    t = traces[0]
    assert "text" in t.input_data
    assert "reasoning_steps" in t.reasoning_data
    assert "confidence_score" in t.reasoning_data


def test_run_loop_prints_conclusion(capsys: object) -> None:
    input_interface = InputInterface()
    engine = ReasoningEngine()
    store = InMemoryStore()

    run_loop(["hello"], input_interface, engine, store)

    captured = capsys.readouterr()  # type: ignore[attr-defined]
    # Conclusion line should be present (deterministic for single-word input)
    assert "Single-word input detected" in captured.out
