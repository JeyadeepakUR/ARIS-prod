"""Tests for deterministic tool abstraction (Module 4)."""

from __future__ import annotations

import uuid
from datetime import UTC, datetime

from aris.input_interface import InputPacket
from aris.memory_store import InMemoryStore, MemoryTrace
from aris.reasoning_engine import ReasoningEngine
from aris.run_loop import run_loop
from aris.tool import EchoTool, ToolRegistry, WordCountTool


class TestTool:
    """Tests for the Tool abstract base class."""

    def test_echo_tool_returns_input(self) -> None:
        tool = EchoTool()
        assert tool.execute("hello") == "hello"
        assert tool.execute("") == ""
        assert tool.execute("multi word input") == "multi word input"

    def test_echo_tool_name(self) -> None:
        tool = EchoTool()
        assert tool.name == "echo"

    def test_word_count_tool_counts_words(self) -> None:
        tool = WordCountTool()
        assert tool.execute("one two three") == "3"
        assert tool.execute("single") == "1"
        assert tool.execute("") == "0"
        assert tool.execute("  spaces   between  ") == "2"

    def test_word_count_tool_name(self) -> None:
        tool = WordCountTool()
        assert tool.name == "word_count"

    def test_tools_are_deterministic(self) -> None:
        echo = EchoTool()
        word_count = WordCountTool()
        input_text = "deterministic output"

        result1_echo = echo.execute(input_text)
        result2_echo = echo.execute(input_text)
        assert result1_echo == result2_echo

        result1_count = word_count.execute(input_text)
        result2_count = word_count.execute(input_text)
        assert result1_count == result2_count


class TestToolRegistry:
    """Tests for ToolRegistry behavior."""

    def test_register_and_get_tool(self) -> None:
        registry = ToolRegistry()
        echo = EchoTool()

        registry.register(echo)
        retrieved = registry.get("echo")

        assert retrieved is echo

    def test_get_nonexistent_tool(self) -> None:
        registry = ToolRegistry()
        result = registry.get("nonexistent")
        assert result is None

    def test_register_multiple_tools(self) -> None:
        registry = ToolRegistry()
        echo = EchoTool()
        word_count = WordCountTool()

        registry.register(echo)
        registry.register(word_count)

        assert registry.get("echo") is echo
        assert registry.get("word_count") is word_count

    def test_overwrite_registered_tool(self) -> None:
        registry = ToolRegistry()
        echo1 = EchoTool()
        echo2 = EchoTool()

        registry.register(echo1)
        registry.register(echo2)

        assert registry.get("echo") is echo2

    def test_list_tools(self) -> None:
        registry = ToolRegistry()
        registry.register(EchoTool())
        registry.register(WordCountTool())

        tools = registry.list_tools()
        assert tools == ["echo", "word_count"]

    def test_empty_registry_list(self) -> None:
        registry = ToolRegistry()
        assert registry.list_tools() == []


class TestRunLoopWithTools:
    """Tests for run_loop tool execution integration."""

    def test_run_loop_without_tools(self) -> None:
        engine = ReasoningEngine()
        store = InMemoryStore()

        packet = InputPacket(
            text="test",
            source="test",
            request_id=uuid.uuid4(),
            timestamp=datetime.now(UTC),
        )

        results = run_loop([packet], engine, store)

        assert len(results) == 1
        traces = store.list_all()
        assert len(traces) == 1
        assert traces[0].tool_calls is None

    def test_run_loop_with_tool_calls(self) -> None:
        engine = ReasoningEngine()
        store = InMemoryStore()
        registry = ToolRegistry()
        registry.register(EchoTool())

        packet = InputPacket(
            text="test",
            source="test",
            request_id=uuid.uuid4(),
            timestamp=datetime.now(UTC),
        )

        tool_calls_map = {0: [("echo", "hello")]}

        results = run_loop([packet], engine, store, registry, tool_calls_map)

        assert len(results) == 1
        traces = store.list_all()
        assert len(traces) == 1
        assert traces[0].tool_calls is not None
        assert len(traces[0].tool_calls) == 1
        assert traces[0].tool_calls[0]["name"] == "echo"
        assert traces[0].tool_calls[0]["input"] == "hello"
        assert traces[0].tool_calls[0]["output"] == "hello"

    def test_run_loop_multiple_tool_calls(self) -> None:
        engine = ReasoningEngine()
        store = InMemoryStore()
        registry = ToolRegistry()
        registry.register(EchoTool())
        registry.register(WordCountTool())

        packet = InputPacket(
            text="test",
            source="test",
            request_id=uuid.uuid4(),
            timestamp=datetime.now(UTC),
        )

        tool_calls_map = {0: [("echo", "hello"), ("word_count", "one two three")]}

        run_loop([packet], engine, store, registry, tool_calls_map)

        traces = store.list_all()
        assert traces[0].tool_calls is not None
        assert len(traces[0].tool_calls) == 2
        assert traces[0].tool_calls[0]["output"] == "hello"
        assert traces[0].tool_calls[1]["output"] == "3"

    def test_run_loop_missing_tool(self) -> None:
        engine = ReasoningEngine()
        store = InMemoryStore()
        registry = ToolRegistry()

        packet = InputPacket(
            text="test",
            source="test",
            request_id=uuid.uuid4(),
            timestamp=datetime.now(UTC),
        )

        tool_calls_map = {0: [("nonexistent", "input")]}

        run_loop([packet], engine, store, registry, tool_calls_map)

        traces = store.list_all()
        assert traces[0].tool_calls == []

    def test_run_loop_tool_calls_deterministic(self) -> None:
        engine = ReasoningEngine()
        registry = ToolRegistry()
        registry.register(WordCountTool())

        packet = InputPacket(
            text="test",
            source="test",
            request_id=uuid.uuid4(),
            timestamp=datetime.now(UTC),
        )

        tool_calls_map = {0: [("word_count", "one two three")]}

        store1 = InMemoryStore()
        store2 = InMemoryStore()

        run_loop([packet], engine, store1, registry, tool_calls_map)
        run_loop([packet], engine, store2, registry, tool_calls_map)

        traces1 = store1.list_all()
        traces2 = store2.list_all()

        assert traces1[0].tool_calls == traces2[0].tool_calls


class TestMemoryTraceToolIntegration:
    """Tests for MemoryTrace tool call recording."""

    def test_memory_trace_without_tool_calls(self) -> None:
        engine = ReasoningEngine()
        packet = InputPacket(
            text="test",
            source="test",
            request_id=uuid.uuid4(),
            timestamp=datetime.now(UTC),
        )
        result = engine.reason(packet)
        trace = MemoryTrace.from_reasoning_result(result)

        assert trace.tool_calls is None

    def test_memory_trace_with_tool_calls(self) -> None:
        engine = ReasoningEngine()
        packet = InputPacket(
            text="test",
            source="test",
            request_id=uuid.uuid4(),
            timestamp=datetime.now(UTC),
        )
        result = engine.reason(packet)
        tool_calls = [
            {"name": "echo", "input": "hello", "output": "hello"},
            {"name": "word_count", "input": "test", "output": "1"},
        ]
        trace = MemoryTrace.from_reasoning_result(result, tool_calls=tool_calls)

        assert trace.tool_calls == tool_calls

    def test_memory_trace_round_trip_with_tools(self) -> None:
        engine = ReasoningEngine()
        packet = InputPacket(
            text="test",
            source="test",
            request_id=uuid.uuid4(),
            timestamp=datetime.now(UTC),
        )
        result = engine.reason(packet)
        tool_calls = [{"name": "echo", "input": "hello", "output": "hello"}]
        original = MemoryTrace.from_reasoning_result(result, tool_calls=tool_calls)

        trace_dict = original.to_dict()
        restored = MemoryTrace.from_dict(trace_dict)

        assert restored.tool_calls == original.tool_calls

    def test_memory_trace_backward_compatibility(self) -> None:
        """Verify traces without tool_calls deserialize correctly."""
        data: dict[str, object] = {
            "request_id": str(uuid.uuid4()),
            "input": {"text": "test", "source": "test"},
            "reasoning": {"reasoning_steps": ["s1"], "confidence_score": 0.5},
            "created_at": datetime.now(UTC).isoformat(),
        }

        trace = MemoryTrace.from_dict(data)
        assert trace.tool_calls is None
