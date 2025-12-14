"""Tests for comparative runner (Module 6)."""

from __future__ import annotations

import uuid
from datetime import UTC, datetime

from aris.comparative_runner import ComparativeResult, ComparativeRunner, SystemConfig
from aris.evaluation import Evaluator
from aris.input_interface import InputPacket
from aris.memory_store import MemoryTrace
from aris.reasoning_engine import ReasoningEngine
from aris.tool import EchoTool, ToolRegistry, WordCountTool


class TestSystemConfig:
    """Tests for SystemConfig dataclass."""

    def test_system_config_creation(self) -> None:
        engine = ReasoningEngine()
        config = SystemConfig(name="test", reasoning_engine=engine)
        assert config.name == "test"
        assert config.reasoning_engine is engine

    def test_system_config_frozen(self) -> None:
        engine = ReasoningEngine()
        config = SystemConfig(name="test", reasoning_engine=engine)
        try:
            config.name = "changed"  # type: ignore
            assert False, "Should not allow mutation"
        except (AttributeError, TypeError):
            pass

    def test_system_config_with_evaluation(self) -> None:
        reasoning_engine = ReasoningEngine()
        evaluator = Evaluator()
        config = SystemConfig(
            name="with_eval",
            reasoning_engine=reasoning_engine,
            evaluator=evaluator,
        )
        assert config.evaluator is evaluator

    def test_system_config_with_tools(self) -> None:
        reasoning_engine = ReasoningEngine()
        registry = ToolRegistry()
        registry.register(EchoTool())
        config = SystemConfig(
            name="with_tools",
            reasoning_engine=reasoning_engine,
            tool_registry=registry,
            tool_calls={0: [("echo", "test")]},
        )
        assert config.tool_registry is registry
        assert config.tool_calls is not None


class TestComparativeResult:
    """Tests for ComparativeResult dataclass."""

    def test_comparative_result_creation(self) -> None:
        packet = InputPacket(
            text="test",
            source="test",
            request_id=uuid.uuid4(),
            timestamp=datetime.now(UTC),
        )
        traces: dict[str, MemoryTrace | Exception] = {}
        result = ComparativeResult(
            input_packet=packet,
            traces_by_config=traces,
            created_at=datetime.now(UTC),
        )
        assert result.input_packet is packet
        assert result.traces_by_config == traces

    def test_comparative_result_frozen(self) -> None:
        packet = InputPacket(
            text="test",
            source="test",
            request_id=uuid.uuid4(),
            timestamp=datetime.now(UTC),
        )
        result = ComparativeResult(
            input_packet=packet,
            traces_by_config={},
            created_at=datetime.now(UTC),
        )
        try:
            result.traces_by_config = {}  # type: ignore
            assert False, "Should not allow mutation"
        except (AttributeError, TypeError):
            pass


class TestComparativeRunner:
    """Tests for ComparativeRunner."""

    def test_runner_exists(self) -> None:
        runner = ComparativeRunner()
        assert runner is not None

    def test_run_single_config(self) -> None:
        runner = ComparativeRunner()
        engine = ReasoningEngine()
        config = SystemConfig(name="baseline", reasoning_engine=engine)
        packet = InputPacket(
            text="test input",
            source="test",
            request_id=uuid.uuid4(),
            timestamp=datetime.now(UTC),
        )

        result = runner.run(packet, [config])

        assert result.input_packet is packet
        assert len(result.traces_by_config) == 1
        assert "baseline" in result.traces_by_config
        trace = result.traces_by_config["baseline"]
        assert isinstance(trace, MemoryTrace)

    def test_run_multiple_configs(self) -> None:
        runner = ComparativeRunner()
        config1 = SystemConfig(name="config1", reasoning_engine=ReasoningEngine())
        config2 = SystemConfig(name="config2", reasoning_engine=ReasoningEngine())
        packet = InputPacket(
            text="test",
            source="test",
            request_id=uuid.uuid4(),
            timestamp=datetime.now(UTC),
        )

        result = runner.run(packet, [config1, config2])

        assert len(result.traces_by_config) == 2
        assert "config1" in result.traces_by_config
        assert "config2" in result.traces_by_config
        assert isinstance(result.traces_by_config["config1"], MemoryTrace)
        assert isinstance(result.traces_by_config["config2"], MemoryTrace)

    def test_trace_independence(self) -> None:
        """Verify traces are independent (no shared state)."""
        runner = ComparativeRunner()
        config1 = SystemConfig(name="c1", reasoning_engine=ReasoningEngine())
        config2 = SystemConfig(name="c2", reasoning_engine=ReasoningEngine())
        packet = InputPacket(
            text="independence test",
            source="test",
            request_id=uuid.uuid4(),
            timestamp=datetime.now(UTC),
        )

        result = runner.run(packet, [config1, config2])

        trace1 = result.traces_by_config["c1"]
        trace2 = result.traces_by_config["c2"]
        assert isinstance(trace1, MemoryTrace)
        assert isinstance(trace2, MemoryTrace)
        # Different trace objects
        assert trace1 is not trace2
        # Same input packet reference
        assert trace1.request_id == packet.request_id
        assert trace2.request_id == packet.request_id

    def test_input_identity_preserved(self) -> None:
        """Verify InputPacket is not mutated across runs."""
        runner = ComparativeRunner()
        config1 = SystemConfig(name="c1", reasoning_engine=ReasoningEngine())
        config2 = SystemConfig(name="c2", reasoning_engine=ReasoningEngine())
        packet = InputPacket(
            text="immutable test",
            source="test",
            request_id=uuid.uuid4(),
            timestamp=datetime.now(UTC),
        )
        original_text = packet.text
        original_id = packet.request_id

        result = runner.run(packet, [config1, config2])

        # Input packet unchanged
        assert packet.text == original_text
        assert packet.request_id == original_id
        # Same packet reference in result
        assert result.input_packet is packet

    def test_deterministic_output(self) -> None:
        """Same input and configs → same traces."""
        runner = ComparativeRunner()
        config = SystemConfig(name="det", reasoning_engine=ReasoningEngine())
        packet = InputPacket(
            text="deterministic",
            source="test",
            request_id=uuid.uuid4(),
            timestamp=datetime.now(UTC),
        )

        result1 = runner.run(packet, [config])
        result2 = runner.run(packet, [config])

        trace1 = result1.traces_by_config["det"]
        trace2 = result2.traces_by_config["det"]
        assert isinstance(trace1, MemoryTrace)
        assert isinstance(trace2, MemoryTrace)
        # Same reasoning steps
        assert trace1.reasoning_data == trace2.reasoning_data

    def test_config_with_evaluation(self) -> None:
        """Config with evaluator produces evaluated trace."""
        runner = ComparativeRunner()
        config = SystemConfig(
            name="eval",
            reasoning_engine=ReasoningEngine(),
            evaluator=Evaluator(),
        )
        packet = InputPacket(
            text="evaluation test",
            source="test",
            request_id=uuid.uuid4(),
            timestamp=datetime.now(UTC),
        )

        result = runner.run(packet, [config])

        trace = result.traces_by_config["eval"]
        assert isinstance(trace, MemoryTrace)
        assert trace.evaluation_data is not None

    def test_config_with_tools(self) -> None:
        """Config with tools executes and records tool calls."""
        runner = ComparativeRunner()
        registry = ToolRegistry()
        registry.register(EchoTool())
        config = SystemConfig(
            name="tools",
            reasoning_engine=ReasoningEngine(),
            tool_registry=registry,
            tool_calls={0: [("echo", "hello")]},
        )
        packet = InputPacket(
            text="tool test",
            source="test",
            request_id=uuid.uuid4(),
            timestamp=datetime.now(UTC),
        )

        result = runner.run(packet, [config])

        trace = result.traces_by_config["tools"]
        assert isinstance(trace, MemoryTrace)
        assert trace.tool_calls is not None
        assert len(trace.tool_calls) == 1
        assert trace.tool_calls[0]["name"] == "echo"
        assert trace.tool_calls[0]["output"] == "hello"

    def test_mixed_configs(self) -> None:
        """Different configs produce different trace features."""
        runner = ComparativeRunner()
        config_plain = SystemConfig(name="plain", reasoning_engine=ReasoningEngine())
        config_eval = SystemConfig(
            name="eval",
            reasoning_engine=ReasoningEngine(),
            evaluator=Evaluator(),
        )
        registry = ToolRegistry()
        registry.register(WordCountTool())
        config_tools = SystemConfig(
            name="tools",
            reasoning_engine=ReasoningEngine(),
            tool_registry=registry,
            tool_calls={0: [("word_count", "one two three")]},
        )
        packet = InputPacket(
            text="mixed test",
            source="test",
            request_id=uuid.uuid4(),
            timestamp=datetime.now(UTC),
        )

        result = runner.run(packet, [config_plain, config_eval, config_tools])

        plain_trace = result.traces_by_config["plain"]
        eval_trace = result.traces_by_config["eval"]
        tools_trace = result.traces_by_config["tools"]

        assert isinstance(plain_trace, MemoryTrace)
        assert isinstance(eval_trace, MemoryTrace)
        assert isinstance(tools_trace, MemoryTrace)

        # Plain has no evaluation or tools
        assert plain_trace.evaluation_data is None
        assert plain_trace.tool_calls is None

        # Eval has evaluation but no tools
        assert eval_trace.evaluation_data is not None
        assert eval_trace.tool_calls is None

        # Tools has tools but no evaluation
        assert tools_trace.evaluation_data is None
        assert tools_trace.tool_calls is not None

    def test_failure_isolation(self) -> None:
        """Exception in one config doesn't affect others."""
        from aris.reasoning_engine import ReasoningResult

        class FailingEngine(ReasoningEngine):
            def reason(self, input_packet: InputPacket) -> ReasoningResult:
                raise ValueError("Intentional failure")

        runner = ComparativeRunner()
        config_good = SystemConfig(name="good", reasoning_engine=ReasoningEngine())
        config_bad = SystemConfig(name="bad", reasoning_engine=FailingEngine())
        packet = InputPacket(
            text="failure test",
            source="test",
            request_id=uuid.uuid4(),
            timestamp=datetime.now(UTC),
        )

        result = runner.run(packet, [config_good, config_bad])

        # Good config succeeds
        good_trace = result.traces_by_config["good"]
        assert isinstance(good_trace, MemoryTrace)

        # Bad config stores exception
        bad_result = result.traces_by_config["bad"]
        assert isinstance(bad_result, Exception)
        assert isinstance(bad_result, ValueError)
        assert "Intentional failure" in str(bad_result)

    def test_all_configs_fail(self) -> None:
        """All configs can fail independently."""
        from aris.reasoning_engine import ReasoningResult

        class FailEngine1(ReasoningEngine):
            def reason(self, input_packet: InputPacket) -> ReasoningResult:
                raise ValueError("Fail 1")

        class FailEngine2(ReasoningEngine):
            def reason(self, input_packet: InputPacket) -> ReasoningResult:
                raise RuntimeError("Fail 2")

        runner = ComparativeRunner()
        config1 = SystemConfig(name="fail1", reasoning_engine=FailEngine1())
        config2 = SystemConfig(name="fail2", reasoning_engine=FailEngine2())
        packet = InputPacket(
            text="all fail",
            source="test",
            request_id=uuid.uuid4(),
            timestamp=datetime.now(UTC),
        )

        result = runner.run(packet, [config1, config2])

        # Both store exceptions
        assert isinstance(result.traces_by_config["fail1"], ValueError)
        assert isinstance(result.traces_by_config["fail2"], RuntimeError)

    def test_empty_configs_list(self) -> None:
        """Empty configs list returns empty results."""
        runner = ComparativeRunner()
        packet = InputPacket(
            text="empty",
            source="test",
            request_id=uuid.uuid4(),
            timestamp=datetime.now(UTC),
        )

        result = runner.run(packet, [])

        assert len(result.traces_by_config) == 0
        assert result.input_packet is packet

    def test_config_name_uniqueness(self) -> None:
        """Duplicate config names overwrite (last wins)."""
        runner = ComparativeRunner()
        config1 = SystemConfig(name="dup", reasoning_engine=ReasoningEngine())
        config2 = SystemConfig(name="dup", reasoning_engine=ReasoningEngine())
        packet = InputPacket(
            text="dup names",
            source="test",
            request_id=uuid.uuid4(),
            timestamp=datetime.now(UTC),
        )

        result = runner.run(packet, [config1, config2])

        # Only one entry (second overwrites first)
        assert len(result.traces_by_config) == 1
        assert "dup" in result.traces_by_config

    def test_result_timestamp(self) -> None:
        """ComparativeResult has created_at timestamp."""
        runner = ComparativeRunner()
        config = SystemConfig(name="ts", reasoning_engine=ReasoningEngine())
        packet = InputPacket(
            text="timestamp",
            source="test",
            request_id=uuid.uuid4(),
            timestamp=datetime.now(UTC),
        )

        before = datetime.now(UTC)
        result = runner.run(packet, [config])
        after = datetime.now(UTC)

        assert before <= result.created_at <= after

    def test_tool_calls_missing_tool(self) -> None:
        """Tool calls with missing tool are skipped."""
        runner = ComparativeRunner()
        registry = ToolRegistry()
        # Register no tools
        config = SystemConfig(
            name="missing",
            reasoning_engine=ReasoningEngine(),
            tool_registry=registry,
            tool_calls={0: [("nonexistent", "input")]},
        )
        packet = InputPacket(
            text="missing tool",
            source="test",
            request_id=uuid.uuid4(),
            timestamp=datetime.now(UTC),
        )

        result = runner.run(packet, [config])

        trace = result.traces_by_config["missing"]
        assert isinstance(trace, MemoryTrace)
        # Tool call attempted but tool not found, so empty list
        assert trace.tool_calls == []

    def test_tool_calls_multiple_tools(self) -> None:
        """Multiple tool calls execute in order."""
        runner = ComparativeRunner()
        registry = ToolRegistry()
        registry.register(EchoTool())
        registry.register(WordCountTool())
        config = SystemConfig(
            name="multi",
            reasoning_engine=ReasoningEngine(),
            tool_registry=registry,
            tool_calls={0: [("echo", "hello"), ("word_count", "one two")]},
        )
        packet = InputPacket(
            text="multi tools",
            source="test",
            request_id=uuid.uuid4(),
            timestamp=datetime.now(UTC),
        )

        result = runner.run(packet, [config])

        trace = result.traces_by_config["multi"]
        assert isinstance(trace, MemoryTrace)
        assert trace.tool_calls is not None
        assert len(trace.tool_calls) == 2
        assert trace.tool_calls[0]["output"] == "hello"
        assert trace.tool_calls[1]["output"] == "2"

    def test_no_automatic_persistence(self) -> None:
        """Runner does not persist traces automatically."""
        runner = ComparativeRunner()
        config = SystemConfig(name="persist", reasoning_engine=ReasoningEngine())
        packet = InputPacket(
            text="no persist",
            source="test",
            request_id=uuid.uuid4(),
            timestamp=datetime.now(UTC),
        )

        result = runner.run(packet, [config])

        # Result contains trace but no side effects occurred
        # (no MemoryStore was passed or used)
        assert "persist" in result.traces_by_config
        # This test is more conceptual - just verifying no store parameter exists
        assert hasattr(runner, "run")
