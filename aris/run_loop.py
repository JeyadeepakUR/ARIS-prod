"""
Deterministic run loop for ARIS.

Provides a testable loop that processes raw user inputs via the InputInterface,
invokes the ReasoningEngine, persists results in a MemoryStore, and prints a
final conclusion for each input.

The loop is deterministic by design:
- No randomness in control flow
- All dependencies are injected (no global state)
- Operates on a provided sequence of raw inputs
"""

from collections.abc import Sequence

from aris.input_interface import InputInterface, InputPacket
from aris.memory_store import MemoryStore, MemoryTrace
from aris.reasoning_engine import ReasoningEngine, ReasoningResult


def _derive_conclusion(result: ReasoningResult) -> str:
    """
    Derive a simple, deterministic conclusion from the reasoning result.

    Currently: returns the last reasoning step as the conclusion.
    This keeps things deterministic and allows LLMs to later replace
    the strategy without changing callers.
    """
    steps = result.reasoning_steps
    return steps[-1] if steps else "No reasoning steps"


def run_loop(
    raw_inputs: Sequence[str],
    input_interface: InputInterface,
    reasoning_engine: ReasoningEngine,
    memory_store: MemoryStore,
    *,
    source: str = "user",
) -> list[ReasoningResult]:
    """
    Process a sequence of raw user inputs deterministically.

    Args:
        raw_inputs: Sequence of raw text inputs to process
        input_interface: InputInterface to transform raw input into InputPacket
        reasoning_engine: ReasoningEngine to generate reasoning
        memory_store: MemoryStore to persist traces

    Returns:
        List of ReasoningResult for each processed input
    """
    results: list[ReasoningResult] = []

    for raw in raw_inputs:
        packet: InputPacket = input_interface.accept(raw, source=source)
        result: ReasoningResult = reasoning_engine.reason(packet)
        trace: MemoryTrace = MemoryTrace.from_reasoning_result(result)
        memory_store.store(trace)

        conclusion = _derive_conclusion(result)
        print(conclusion)

        results.append(result)

    return results
