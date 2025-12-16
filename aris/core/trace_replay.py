"""
Deterministic trace replay engine for ARIS.

Reconstructs execution flows from memory traces without recomputation or mutation.
Emits immutable replay frames in strict ordering: input → reasoning steps →
evaluation → tool calls. Skips absent sections gracefully.
"""

import uuid
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Literal

from aris.core.memory_store import MemoryTrace


@dataclass(frozen=True)
class ReplayFrame:
    """
    Immutable record of a single event in trace replay.

    Attributes:
        frame_index: Position in the replay sequence (0-indexed)
        frame_type: Category of event (input, reasoning_step, evaluation, tool_call)
        content: Dictionary with event data (structure depends on frame_type)
    """

    frame_index: int
    frame_type: Literal["input", "reasoning_step", "evaluation", "tool_call"]
    content: dict[str, object]

    def to_dict(self) -> dict[str, object]:
        """Convert frame to JSON-serializable dictionary."""
        return {
            "frame_index": self.frame_index,
            "frame_type": self.frame_type,
            "content": self.content,
        }

    @classmethod
    def from_dict(cls, data: dict[str, object]) -> "ReplayFrame":
        """Reconstruct frame from dictionary."""
        content_obj = data["content"]
        if not isinstance(content_obj, dict):
            raise TypeError("content field must be a dictionary")

        frame_index_obj = data["frame_index"]
        if not isinstance(frame_index_obj, int):
            raise TypeError("frame_index must be an integer")

        return cls(
            frame_index=frame_index_obj,
            frame_type=data["frame_type"],  # type: ignore
            content={k: v for k, v in content_obj.items()},
        )


@dataclass(frozen=True)
class TraceReplay:
    """
    Immutable trace replay with ordered frames.

    Represents the complete execution flow reconstructed from a MemoryTrace,
    with frames ordered sequentially: input, reasoning steps, evaluation, tool calls.

    Attributes:
        request_id: Identifier from source trace
        frames: Ordered list of ReplayFrame objects
        created_at: UTC timestamp when replay was created
    """

    request_id: uuid.UUID
    frames: list[ReplayFrame]
    created_at: datetime

    def to_dict(self) -> dict[str, object]:
        """Convert replay to JSON-serializable dictionary."""
        return {
            "request_id": str(self.request_id),
            "frames": [f.to_dict() for f in self.frames],
            "created_at": self.created_at.isoformat(),
        }

    @classmethod
    def from_dict(cls, data: dict[str, object]) -> "TraceReplay":
        """Reconstruct replay from dictionary."""
        frames_obj = data["frames"]
        if not isinstance(frames_obj, list):
            raise TypeError("frames field must be a list")

        frames: list[ReplayFrame] = []
        for frame_data in frames_obj:
            if not isinstance(frame_data, dict):
                raise TypeError("each frame must be a dictionary")
            frames.append(ReplayFrame.from_dict(frame_data))

        return cls(
            request_id=uuid.UUID(str(data["request_id"])),
            frames=frames,
            created_at=datetime.fromisoformat(str(data["created_at"])),
        )


class ReplayEngine:
    """
    Deterministic trace replay engine.

    Reconstructs execution flow from MemoryTrace without recomputation.
    Emits immutable frames in strict ordering without fabricating missing data.
    """

    def replay(self, trace: MemoryTrace) -> TraceReplay:
        """
        Replay a memory trace into ordered frames.

        Reconstruction order:
        1. Input frame (from input_data)
        2. Reasoning frames (one per reasoning step)
        3. Evaluation frame (if evaluation_data present)
        4. Tool call frames (one per tool call)

        Args:
            trace: MemoryTrace to replay

        Returns:
            TraceReplay with frames in strict ordering
        """
        frames: list[ReplayFrame] = []
        frame_index = 0

        # Frame 0: Input
        input_content: dict[str, object] = {
            k: v for k, v in trace.input_data.items()
        }
        input_frame = ReplayFrame(
            frame_index=frame_index,
            frame_type="input",
            content=input_content,
        )
        frames.append(input_frame)
        frame_index += 1

        # Frames 1+: Reasoning steps
        reasoning_steps = trace.reasoning_data.get("reasoning_steps", [])
        if isinstance(reasoning_steps, list):
            for step_idx, step in enumerate(reasoning_steps):
                reasoning_frame = ReplayFrame(
                    frame_index=frame_index,
                    frame_type="reasoning_step",
                    content={
                        "step_index": step_idx,
                        "step": step,
                    },
                )
                frames.append(reasoning_frame)
                frame_index += 1

        # Add confidence score from reasoning to context (not a separate frame)
        confidence = trace.reasoning_data.get("confidence_score")
        if confidence is not None:
            # Store confidence as metadata (could be accessed from reasoning_data)
            pass

        # Evaluation frame (if present)
        if trace.evaluation_data is not None:
            eval_content: dict[str, object] = {
                k: v for k, v in trace.evaluation_data.items()
            }
            evaluation_frame = ReplayFrame(
                frame_index=frame_index,
                frame_type="evaluation",
                content=eval_content,
            )
            frames.append(evaluation_frame)
            frame_index += 1

        # Tool call frames (if present)
        if trace.tool_calls is not None:
            for tool_call_idx, tool_call in enumerate(trace.tool_calls):
                tool_call_frame = ReplayFrame(
                    frame_index=frame_index,
                    frame_type="tool_call",
                    content={
                        "tool_call_index": tool_call_idx,
                        "name": tool_call.get("name"),
                        "input": tool_call.get("input"),
                        "output": tool_call.get("output"),
                    },
                )
                frames.append(tool_call_frame)
                frame_index += 1

        return TraceReplay(
            request_id=trace.request_id,
            frames=frames,
            created_at=datetime.now(UTC),
        )
