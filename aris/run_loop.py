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

from aris.input_interface import InputPacket
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
    input_packets: Sequence[InputPacket],
    reasoning_engine: ReasoningEngine,
    memory_store: MemoryStore,
) -> list[ReasoningResult]:
    """
    Process a sequence of validated input packets deterministically.

    Args:
        input_packets: Validated InputPacket objects to reason over
        reasoning_engine: ReasoningEngine to generate reasoning
        memory_store: MemoryStore to persist traces

    Returns:
        List of ReasoningResult for each processed input
    """
    results: list[ReasoningResult] = []

    for packet in input_packets:
        result: ReasoningResult = reasoning_engine.reason(packet)
        trace: MemoryTrace = MemoryTrace.from_reasoning_result(result)
        memory_store.store(trace)

        conclusion = _derive_conclusion(result)
        print(conclusion)

        results.append(result)

    return results
