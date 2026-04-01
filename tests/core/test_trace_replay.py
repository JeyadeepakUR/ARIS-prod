"""Tests for trace replay engine (Module 5)."""

from __future__ import annotations

import json
import uuid
from datetime import UTC, datetime

from aris.core.memory_store import MemoryTrace
from aris.core.trace_replay import ReplayEngine, ReplayFrame, TraceReplay


class TestReplayFrame:
    """Tests for ReplayFrame dataclass."""

    def test_replay_frame_creation(self) -> None:
        frame = ReplayFrame(
            frame_index=0,
            frame_type="input",
            content={"text": "test"},
        )
        assert frame.frame_index == 0
        assert frame.frame_type == "input"
        assert frame.content == {"text": "test"}

    def test_replay_frame_frozen(self) -> None:
        frame = ReplayFrame(
            frame_index=0,
            frame_type="input",
            content={"text": "test"},
        )
        try:
            frame.frame_index = 1  # type: ignore
            assert False, "Should not allow mutation"
        except (AttributeError, TypeError):
            pass

    def test_replay_frame_to_dict(self) -> None:
        frame = ReplayFrame(
            frame_index=0,
            frame_type="reasoning_step",
            content={"step_index": 0, "step": "reason"},
        )
        data = frame.to_dict()
        assert data["frame_index"] == 0
        assert data["frame_type"] == "reasoning_step"
        content_obj = data["content"]
        if isinstance(content_obj, dict):
            assert content_obj["step"] == "reason"

    def test_replay_frame_from_dict(self) -> None:
        data: dict[str, object] = {
            "frame_index": 1,
            "frame_type": "tool_call",
            "content": {"name": "echo", "input": "x", "output": "x"},
        }
        frame = ReplayFrame.from_dict(data)
        assert frame.frame_index == 1
        assert frame.frame_type == "tool_call"
        content_name = frame.content.get("name")
        assert content_name == "echo"

    def test_replay_frame_round_trip(self) -> None:
        original = ReplayFrame(
            frame_index=2,
            frame_type="evaluation",
            content={"score": 0.9},
        )
        original_dict = original.to_dict()
        restored = ReplayFrame.from_dict(original_dict)
        assert restored.frame_index == original.frame_index
        assert restored.frame_type == original.frame_type
        assert restored.content == original.content


class TestTraceReplay:
    """Tests for TraceReplay dataclass."""

    def test_trace_replay_creation(self) -> None:
        req_id = uuid.uuid4()
        frames = [
            ReplayFrame(0, "input", {"text": "test"}),
            ReplayFrame(1, "reasoning_step", {"step": "s1"}),
        ]
        replay = TraceReplay(
            request_id=req_id,
            frames=frames,
            created_at=datetime.now(UTC),
        )
        assert replay.request_id == req_id
        assert len(replay.frames) == 2

    def test_trace_replay_frozen(self) -> None:
        req_id = uuid.uuid4()
        frames = [ReplayFrame(0, "input", {"text": "test"})]
        replay = TraceReplay(
            request_id=req_id,
            frames=frames,
            created_at=datetime.now(UTC),
        )
        try:
            replay.frames = []  # type: ignore
            assert False, "Should not allow mutation"
        except (AttributeError, TypeError):
            pass

    def test_trace_replay_to_dict(self) -> None:
        req_id = uuid.uuid4()
        frames = [
            ReplayFrame(0, "input", {"text": "test"}),
            ReplayFrame(1, "reasoning_step", {"step": "s1"}),
        ]
        replay = TraceReplay(
            request_id=req_id,
            frames=frames,
            created_at=datetime.now(UTC),
        )
        data = replay.to_dict()
        assert str(req_id) == data["request_id"]
        frames_obj = data["frames"]
        if isinstance(frames_obj, list):
            assert len(frames_obj) == 2
            first_frame = frames_obj[0]
            if isinstance(first_frame, dict):
                assert first_frame["frame_type"] == "input"

    def test_trace_replay_from_dict(self) -> None:
        req_id = uuid.uuid4()
        now = datetime.now(UTC)
        data: dict[str, object] = {
            "request_id": str(req_id),
            "frames": [
                {
                    "frame_index": 0,
                    "frame_type": "input",
                    "content": {"text": "test"},
                },
            ],
            "created_at": now.isoformat(),
        }
        replay = TraceReplay.from_dict(data)
        assert replay.request_id == req_id
        assert len(replay.frames) == 1

    def test_trace_replay_round_trip(self) -> None:
        req_id = uuid.uuid4()
        original = TraceReplay(
            request_id=req_id,
            frames=[
                ReplayFrame(0, "input", {"text": "test"}),
                ReplayFrame(1, "reasoning_step", {"step": "s1"}),
                ReplayFrame(2, "evaluation", {"score": 0.9}),
            ],
            created_at=datetime.now(UTC),
        )
        restored = TraceReplay.from_dict(original.to_dict())
        assert restored.request_id == original.request_id
        assert len(restored.frames) == len(original.frames)
        assert restored.frames[0].frame_type == "input"
        assert restored.frames[1].frame_type == "reasoning_step"
        assert restored.frames[2].frame_type == "evaluation"

    def test_trace_replay_json_serializable(self) -> None:
        req_id = uuid.uuid4()
        replay = TraceReplay(
            request_id=req_id,
            frames=[
                ReplayFrame(0, "input", {"text": "test"}),
                ReplayFrame(1, "reasoning_step", {"step": "s1"}),
            ],
            created_at=datetime.now(UTC),
        )
        data = replay.to_dict()
        json_str = json.dumps(data)
        assert json_str is not None
        restored_data = json.loads(json_str)
        assert restored_data["request_id"] == str(req_id)


class TestReplayEngine:
    """Tests for ReplayEngine."""

    def test_replay_engine_exists(self) -> None:
        engine = ReplayEngine()
        assert engine is not None

    def test_replay_input_only(self) -> None:
        """Replay trace with input only (minimal trace)."""
        engine = ReplayEngine()
        req_id = uuid.uuid4()
        trace = MemoryTrace(
            request_id=req_id,
            input_data={"text": "test", "source": "test"},
            reasoning_data={"reasoning_steps": [], "confidence_score": 0.5},
            created_at=datetime.now(UTC),
        )

        replay = engine.replay(trace)

        assert replay.request_id == req_id
        assert len(replay.frames) == 1
        assert replay.frames[0].frame_type == "input"
        assert replay.frames[0].content["text"] == "test"

    def test_replay_with_reasoning_steps(self) -> None:
        """Replay with input and reasoning steps."""
        engine = ReplayEngine()
        req_id = uuid.uuid4()
        trace = MemoryTrace(
            request_id=req_id,
            input_data={"text": "test", "source": "test"},
            reasoning_data={
                "reasoning_steps": ["step1", "step2", "step3"],
                "confidence_score": 0.8,
            },
            created_at=datetime.now(UTC),
        )

        replay = engine.replay(trace)

        assert len(replay.frames) == 4  # input + 3 reasoning steps
        assert replay.frames[0].frame_type == "input"
        assert replay.frames[1].frame_type == "reasoning_step"
        assert replay.frames[1].content["step"] == "step1"
        assert replay.frames[2].content["step"] == "step2"
        assert replay.frames[3].content["step"] == "step3"

    def test_replay_with_evaluation(self) -> None:
        """Replay with evaluation section."""
        engine = ReplayEngine()
        req_id = uuid.uuid4()
        trace = MemoryTrace(
            request_id=req_id,
            input_data={"text": "test", "source": "test"},
            reasoning_data={
                "reasoning_steps": ["step1"],
                "confidence_score": 0.7,
            },
            evaluation_data={
                "step_count_score": 0.5,
                "coherence_score": 0.9,
                "confidence_alignment_score": 0.7,
                "input_coverage_score": 0.8,
                "overall_score": 0.75,
            },
            created_at=datetime.now(UTC),
        )

        replay = engine.replay(trace)

        # Frames: input, reasoning_step, evaluation
        assert len(replay.frames) == 3
        assert replay.frames[2].frame_type == "evaluation"
        assert replay.frames[2].content["overall_score"] == 0.75

    def test_replay_with_tool_calls(self) -> None:
        """Replay with tool calls."""
        engine = ReplayEngine()
        req_id = uuid.uuid4()
        trace = MemoryTrace(
            request_id=req_id,
            input_data={"text": "test", "source": "test"},
            reasoning_data={
                "reasoning_steps": ["step1"],
                "confidence_score": 0.8,
            },
            tool_calls=[
                {"name": "echo", "input": "hello", "output": "hello"},
                {"name": "word_count", "input": "test", "output": "1"},
            ],
            created_at=datetime.now(UTC),
        )

        replay = engine.replay(trace)

        # Frames: input, reasoning_step, tool_call, tool_call
        assert len(replay.frames) == 4
        assert replay.frames[2].frame_type == "tool_call"
        assert replay.frames[2].content["name"] == "echo"
        assert replay.frames[3].content["name"] == "word_count"

    def test_replay_full_trace(self) -> None:
        """Replay with all sections: input, reasoning, evaluation, tools."""
        engine = ReplayEngine()
        req_id = uuid.uuid4()
        trace = MemoryTrace(
            request_id=req_id,
            input_data={"text": "full test", "source": "test"},
            reasoning_data={
                "reasoning_steps": ["analyze", "decide"],
                "confidence_score": 0.85,
            },
            evaluation_data={
                "step_count_score": 0.8,
                "coherence_score": 0.9,
                "confidence_alignment_score": 0.85,
                "input_coverage_score": 0.9,
                "overall_score": 0.86,
            },
            tool_calls=[
                {"name": "echo", "input": "x", "output": "x"},
            ],
            created_at=datetime.now(UTC),
        )

        replay = engine.replay(trace)

        # Frames: input (1) + reasoning (2) + evaluation (1) + tools (1) = 5
        assert len(replay.frames) == 5
        assert replay.frames[0].frame_type == "input"
        assert replay.frames[1].frame_type == "reasoning_step"
        assert replay.frames[2].frame_type == "reasoning_step"
        assert replay.frames[3].frame_type == "evaluation"
        assert replay.frames[4].frame_type == "tool_call"

    def test_replay_frame_ordering(self) -> None:
        """Verify strict frame ordering: input → reasoning → eval → tools."""
        engine = ReplayEngine()
        req_id = uuid.uuid4()
        trace = MemoryTrace(
            request_id=req_id,
            input_data={"text": "order test", "source": "test"},
            reasoning_data={
                "reasoning_steps": ["s1", "s2"],
                "confidence_score": 0.9,
            },
            evaluation_data={"overall_score": 0.9},
            tool_calls=[
                {"name": "t1", "input": "i1", "output": "o1"},
                {"name": "t2", "input": "i2", "output": "o2"},
            ],
            created_at=datetime.now(UTC),
        )

        replay = engine.replay(trace)

        # Verify sequence strictly
        frame_types = [f.frame_type for f in replay.frames]
        expected = [
            "input",
            "reasoning_step",
            "reasoning_step",
            "evaluation",
            "tool_call",
            "tool_call",
        ]
        assert frame_types == expected

    def test_replay_frame_indices_sequential(self) -> None:
        """Verify frame indices are sequential from 0."""
        engine = ReplayEngine()
        req_id = uuid.uuid4()
        trace = MemoryTrace(
            request_id=req_id,
            input_data={"text": "test", "source": "test"},
            reasoning_data={
                "reasoning_steps": ["s1", "s2"],
                "confidence_score": 0.8,
            },
            tool_calls=[
                {"name": "echo", "input": "x", "output": "x"},
            ],
            created_at=datetime.now(UTC),
        )

        replay = engine.replay(trace)

        frame_indices = [f.frame_index for f in replay.frames]
        assert frame_indices == list(range(len(replay.frames)))

    def test_replay_deterministic(self) -> None:
        """Same trace → same replay output."""
        engine = ReplayEngine()
        req_id = uuid.uuid4()
        trace = MemoryTrace(
            request_id=req_id,
            input_data={"text": "deterministic", "source": "test"},
            reasoning_data={
                "reasoning_steps": ["s1", "s2", "s3"],
                "confidence_score": 0.75,
            },
            evaluation_data={"overall_score": 0.8},
            tool_calls=[
                {"name": "word_count", "input": "one two", "output": "2"},
            ],
            created_at=datetime.now(UTC),
        )

        replay1 = engine.replay(trace)
        replay2 = engine.replay(trace)

        # Compare frame counts and types
        assert len(replay1.frames) == len(replay2.frames)
        for f1, f2 in zip(replay1.frames, replay2.frames):
            assert f1.frame_index == f2.frame_index
            assert f1.frame_type == f2.frame_type
            assert f1.content == f2.content

    def test_replay_no_mutation(self) -> None:
        """Replay does not mutate original trace."""
        engine = ReplayEngine()
        req_id = uuid.uuid4()
        original_input = {"text": "test", "source": "test"}
        trace = MemoryTrace(
            request_id=req_id,
            input_data=original_input,
            reasoning_data={
                "reasoning_steps": ["s1"],
                "confidence_score": 0.8,
            },
            created_at=datetime.now(UTC),
        )

        engine.replay(trace)

        # Verify trace unchanged
        assert trace.input_data == original_input

    def test_replay_backward_compat_no_evaluation(self) -> None:
        """Replay handles traces without evaluation section."""
        engine = ReplayEngine()
        req_id = uuid.uuid4()
        trace = MemoryTrace(
            request_id=req_id,
            input_data={"text": "test", "source": "test"},
            reasoning_data={
                "reasoning_steps": ["s1"],
                "confidence_score": 0.8,
            },
            evaluation_data=None,
            created_at=datetime.now(UTC),
        )

        replay = engine.replay(trace)

        # Should have input + reasoning, no evaluation frame
        assert len(replay.frames) == 2
        assert replay.frames[1].frame_type == "reasoning_step"

    def test_replay_backward_compat_no_tools(self) -> None:
        """Replay handles traces without tool_calls section."""
        engine = ReplayEngine()
        req_id = uuid.uuid4()
        trace = MemoryTrace(
            request_id=req_id,
            input_data={"text": "test", "source": "test"},
            reasoning_data={
                "reasoning_steps": ["s1"],
                "confidence_score": 0.8,
            },
            tool_calls=None,
            created_at=datetime.now(UTC),
        )

        replay = engine.replay(trace)

        # Should have input + reasoning, no tool frames
        assert len(replay.frames) == 2
        frame_types = [f.frame_type for f in replay.frames]
        assert "tool_call" not in frame_types

    def test_replay_empty_reasoning_steps(self) -> None:
        """Replay handles empty reasoning_steps list."""
        engine = ReplayEngine()
        req_id = uuid.uuid4()
        trace = MemoryTrace(
            request_id=req_id,
            input_data={"text": "test", "source": "test"},
            reasoning_data={
                "reasoning_steps": [],
                "confidence_score": 0.5,
            },
            created_at=datetime.now(UTC),
        )

        replay = engine.replay(trace)

        # Only input frame, no reasoning frames
        assert len(replay.frames) == 1
        assert replay.frames[0].frame_type == "input"

    def test_replay_empty_tool_calls(self) -> None:
        """Replay handles empty tool_calls list."""
        engine = ReplayEngine()
        req_id = uuid.uuid4()
        trace = MemoryTrace(
            request_id=req_id,
            input_data={"text": "test", "source": "test"},
            reasoning_data={
                "reasoning_steps": ["s1"],
                "confidence_score": 0.8,
            },
            tool_calls=[],
            created_at=datetime.now(UTC),
        )

        replay = engine.replay(trace)

        # Input + reasoning, no tool frames (empty list means no tools)
        assert len(replay.frames) == 2
        frame_types = [f.frame_type for f in replay.frames]
        assert "tool_call" not in frame_types

    def test_replay_reasoning_step_indices(self) -> None:
        """Verify step_index in reasoning frames."""
        engine = ReplayEngine()
        req_id = uuid.uuid4()
        trace = MemoryTrace(
            request_id=req_id,
            input_data={"text": "test", "source": "test"},
            reasoning_data={
                "reasoning_steps": ["a", "b", "c"],
                "confidence_score": 0.8,
            },
            created_at=datetime.now(UTC),
        )

        replay = engine.replay(trace)

        # Check step indices
        assert replay.frames[1].content["step_index"] == 0
        assert replay.frames[2].content["step_index"] == 1
        assert replay.frames[3].content["step_index"] == 2

    def test_replay_tool_call_indices(self) -> None:
        """Verify tool_call_index in tool frames."""
        engine = ReplayEngine()
        req_id = uuid.uuid4()
        trace = MemoryTrace(
            request_id=req_id,
            input_data={"text": "test", "source": "test"},
            reasoning_data={
                "reasoning_steps": ["s1"],
                "confidence_score": 0.8,
            },
            tool_calls=[
                {"name": "echo", "input": "a", "output": "a"},
                {"name": "echo", "input": "b", "output": "b"},
            ],
            created_at=datetime.now(UTC),
        )

        replay = engine.replay(trace)

        # Check tool_call indices
        assert replay.frames[2].content["tool_call_index"] == 0
        assert replay.frames[3].content["tool_call_index"] == 1

    def test_replay_complex_content(self) -> None:
        """Verify nested content structures are preserved."""
        engine = ReplayEngine()
        req_id = uuid.uuid4()
        trace = MemoryTrace(
            request_id=req_id,
            input_data={
                "text": "complex",
                "source": "test",
                "metadata": "value",
            },
            reasoning_data={
                "reasoning_steps": ["step"],
                "confidence_score": 0.9,
                "extra": "data",
            },
            created_at=datetime.now(UTC),
        )

        replay = engine.replay(trace)

        # Verify input content includes all fields
        input_content = replay.frames[0].content
        assert "text" in input_content
        assert "source" in input_content
        assert "metadata" in input_content

    def test_replay_from_real_trace(self) -> None:
        """Replay a trace built from actual reasoning engine."""
        from aris.core.input_interface import InputPacket
        from aris.core.reasoning_engine import ReasoningEngine as EngineForReasoning

        engine = ReplayEngine()
        reasoning_engine = EngineForReasoning()

        packet = InputPacket(
            text="real trace test",
            source="test",
            request_id=uuid.uuid4(),
            timestamp=datetime.now(UTC),
        )

        result = reasoning_engine.reason(packet)
        trace = MemoryTrace.from_reasoning_result(result)

        replay = engine.replay(trace)

        # Should have at least input + reasoning steps
        assert len(replay.frames) >= 2
        assert replay.frames[0].frame_type == "input"
        assert replay.frames[1].frame_type == "reasoning_step"
