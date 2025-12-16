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

from aris.core.input_interface import InputPacket
from aris.core.memory_store import MemoryStore, MemoryTrace
from aris.core.reasoning_engine import ReasoningEngine, ReasoningResult
from aris.core.tool import ToolRegistry


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
    tool_registry: ToolRegistry | None = None,
    tool_calls: dict[int, list[tuple[str, str]]] | None = None,
) -> list[ReasoningResult]:
    """
    Process a sequence of validated input packets deterministically.

    Args:
        input_packets: Validated InputPacket objects to reason over
        reasoning_engine: ReasoningEngine to generate reasoning
        memory_store: MemoryStore to persist traces
        tool_registry: Optional registry of tools available for execution
        tool_calls: Optional mapping of packet index to list of (tool_name, input) tuples

    Returns:
        List of ReasoningResult for each processed input
    """
    results: list[ReasoningResult] = []

    for idx, packet in enumerate(input_packets):
        result: ReasoningResult = reasoning_engine.reason(packet)

        # Optionally execute tool calls
        tool_call_records: list[dict[str, str]] | None = None
        if tool_registry is not None and tool_calls is not None and idx in tool_calls:
            tool_call_records = []
            for tool_name, tool_input in tool_calls[idx]:
                tool = tool_registry.get(tool_name)
                if tool is not None:
                    tool_output = tool.execute(tool_input)
                    tool_call_records.append(
                        {
                            "name": tool_name,
                            "input": tool_input,
                            "output": tool_output,
                        }
                    )

        trace: MemoryTrace = MemoryTrace.from_reasoning_result(
            result, tool_calls=tool_call_records
        )
        memory_store.store(trace)

        conclusion = _derive_conclusion(result)
        print(conclusion)

        results.append(result)

    return results
