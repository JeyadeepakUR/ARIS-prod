"""
Comparative runner for testing system variants in ARIS.

Executes the same InputPacket across multiple system configurations,
producing independent MemoryTrace objects without merging or aggregation.
"""

from dataclasses import dataclass
from datetime import UTC, datetime

from aris.evaluation import EvaluationResult, Evaluator
from aris.input_interface import InputPacket
from aris.memory_store import MemoryTrace
from aris.reasoning_engine import ReasoningEngine, ReasoningResult
from aris.tool import ToolRegistry


@dataclass(frozen=True)
class SystemConfig:
    """
    Configuration describing a system variant.

    Attributes:
        name: Unique identifier for this configuration
        reasoning_engine: ReasoningEngine instance to use
        evaluator: Optional Evaluator for scoring
        tool_registry: Optional ToolRegistry for tool execution
        tool_calls: Optional mapping of packet index to tool calls
    """

    name: str
    reasoning_engine: ReasoningEngine
    evaluator: Evaluator | None = None
    tool_registry: ToolRegistry | None = None
    tool_calls: dict[int, list[tuple[str, str]]] | None = None


@dataclass(frozen=True)
class ComparativeResult:
    """
    Results from running the same input across multiple configurations.

    Attributes:
        input_packet: The InputPacket that was executed
        traces_by_config: Mapping of config name to MemoryTrace or Exception
        created_at: UTC timestamp when the comparison was run
    """

    input_packet: InputPacket
    traces_by_config: dict[str, MemoryTrace | Exception]
    created_at: datetime


class ComparativeRunner:
    """
    Executes the same InputPacket across multiple system configurations.

    Produces independent MemoryTrace objects per configuration without
    merging, persistence, or aggregation. Isolates failures per configuration.
    """

    def run(
        self,
        input_packet: InputPacket,
        configs: list[SystemConfig],
    ) -> ComparativeResult:
        """
        Execute input_packet across all configurations.

        For each configuration:
        1. Run reasoning engine
        2. Optionally run evaluation engine
        3. Optionally execute tool calls
        4. Create MemoryTrace
        5. Catch and store any exceptions

        Args:
            input_packet: InputPacket to execute (not mutated)
            configs: List of SystemConfig variants to test

        Returns:
            ComparativeResult with traces or exceptions per config
        """
        traces_by_config: dict[str, MemoryTrace | Exception] = {}

        for config in configs:
            try:
                # Run reasoning engine
                reasoning_result: ReasoningResult = config.reasoning_engine.reason(
                    input_packet
                )

                # Optionally run evaluation
                evaluation_result: EvaluationResult | None = None
                if config.evaluator is not None:
                    evaluation_result = config.evaluator.evaluate(reasoning_result)

                # Optionally execute tool calls
                tool_call_records: list[dict[str, str]] | None = None
                if (
                    config.tool_registry is not None
                    and config.tool_calls is not None
                    and 0 in config.tool_calls
                ):
                    tool_call_records = []
                    for tool_name, tool_input in config.tool_calls[0]:
                        tool = config.tool_registry.get(tool_name)
                        if tool is not None:
                            tool_output = tool.execute(tool_input)
                            tool_call_records.append(
                                {
                                    "name": tool_name,
                                    "input": tool_input,
                                    "output": tool_output,
                                }
                            )

                # Create memory trace
                trace = MemoryTrace.from_reasoning_result(
                    reasoning_result,
                    evaluation=evaluation_result,
                    tool_calls=tool_call_records,
                )

                traces_by_config[config.name] = trace

            except Exception as e:
                # Isolate failures per configuration
                traces_by_config[config.name] = e

        return ComparativeResult(
            input_packet=input_packet,
            traces_by_config=traces_by_config,
            created_at=datetime.now(UTC),
        )
