"""
Memory store abstraction for persisting reasoning traces.

This module provides a thread-safe abstraction for storing and retrieving
reasoning traces, with both in-memory and file-backed implementations.
"""

import json
import threading
import uuid
from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

from aris.reasoning_engine import ReasoningResult


@dataclass(frozen=True)
class MemoryTrace:
    """
    Immutable record of a complete reasoning trace.

    Attributes:
        request_id: Unique identifier for this trace (from input packet)
        input_data: Dictionary representation of the input packet
        reasoning_data: Dictionary representation of the reasoning result
        created_at: UTC timestamp when the trace was created
    """

    request_id: uuid.UUID
    input_data: dict[str, str]
    reasoning_data: dict[str, object]
    created_at: datetime

    @classmethod
    def from_reasoning_result(cls, result: ReasoningResult) -> "MemoryTrace":
        """
        Create a MemoryTrace from a ReasoningResult.

        Args:
            result: ReasoningResult to convert to a trace

        Returns:
            MemoryTrace containing the reasoning result data
        """
        input_packet = result.input_packet

        input_data = {
            "text": input_packet.text,
            "request_id": str(input_packet.request_id),
            "timestamp": input_packet.timestamp.isoformat(),
        }

        reasoning_data = {
            "reasoning_steps": result.reasoning_steps,
            "confidence_score": result.confidence_score,
        }

        return cls(
            request_id=input_packet.request_id,
            input_data=input_data,
            reasoning_data=reasoning_data,
            created_at=datetime.now(UTC),
        )

    def to_dict(self) -> dict[str, object]:
        """
        Convert trace to dictionary format matching the required schema.

        Returns:
            Dictionary with request_id, input, reasoning, and created_at fields
        """
        return {
            "request_id": str(self.request_id),
            "input": self.input_data,
            "reasoning": self.reasoning_data,
            "created_at": self.created_at.isoformat(),
        }

    @classmethod
    def from_dict(cls, data: dict[str, object]) -> "MemoryTrace":
        """
        Create a MemoryTrace from dictionary format.

        Args:
            data: Dictionary with trace data

        Returns:
            MemoryTrace instance
        """
        # Type narrowing for dict access
        input_obj = data["input"]
        reasoning_obj = data["reasoning"]

        if not isinstance(input_obj, dict):
            raise TypeError("input field must be a dictionary")
        if not isinstance(reasoning_obj, dict):
            raise TypeError("reasoning field must be a dictionary")

        return cls(
            request_id=uuid.UUID(str(data["request_id"])),
            input_data=input_obj,
            reasoning_data=reasoning_obj,
            created_at=datetime.fromisoformat(str(data["created_at"])),
        )


class MemoryStore(ABC):
    """
    Abstract base class for memory storage implementations.

    All implementations must be thread-safe and support storing/retrieving
    reasoning traces by request ID.
    """

    @abstractmethod
    def store(self, trace: MemoryTrace) -> None:
        """
        Store a memory trace.

        Args:
            trace: MemoryTrace to store
        """
        pass

    @abstractmethod
    def retrieve(self, request_id: uuid.UUID) -> MemoryTrace | None:
        """
        Retrieve a memory trace by request ID.

        Args:
            request_id: UUID of the trace to retrieve

        Returns:
            MemoryTrace if found, None otherwise
        """
        pass

    @abstractmethod
    def list_all(self) -> list[MemoryTrace]:
        """
        List all stored memory traces.

        Returns:
            List of all MemoryTrace objects in the store
        """
        pass

    @abstractmethod
    def clear(self) -> None:
        """
        Clear all stored traces.

        Used primarily for testing and cleanup.
        """
        pass


class InMemoryStore(MemoryStore):
    """
    Thread-safe in-memory implementation of MemoryStore.

    Stores traces in a dictionary for fast access. Suitable for testing
    and scenarios where persistence is not required.
    """

    def __init__(self) -> None:
        """Initialize the in-memory store with an empty dictionary and lock."""
        self._traces: dict[uuid.UUID, MemoryTrace] = {}
        self._lock = threading.RLock()

    def store(self, trace: MemoryTrace) -> None:
        """
        Store a memory trace in memory.

        Args:
            trace: MemoryTrace to store
        """
        with self._lock:
            self._traces[trace.request_id] = trace

    def retrieve(self, request_id: uuid.UUID) -> MemoryTrace | None:
        """
        Retrieve a memory trace by request ID.

        Args:
            request_id: UUID of the trace to retrieve

        Returns:
            MemoryTrace if found, None otherwise
        """
        with self._lock:
            return self._traces.get(request_id)

    def list_all(self) -> list[MemoryTrace]:
        """
        List all stored memory traces.

        Returns:
            List of all MemoryTrace objects sorted by created_at
        """
        with self._lock:
            return sorted(self._traces.values(), key=lambda t: t.created_at)

    def clear(self) -> None:
        """Clear all stored traces."""
        with self._lock:
            self._traces.clear()


class FileBackedMemoryStore(MemoryStore):
    """
    Thread-safe file-backed JSON implementation of MemoryStore.

    Stores traces in individual JSON files within a specified directory.
    Each trace is stored as a separate file named by its request ID.
    """

    def __init__(self, storage_dir: Path) -> None:
        """
        Initialize the file-backed store.

        Args:
            storage_dir: Directory path where trace files will be stored
        """
        self._storage_dir = storage_dir
        self._lock = threading.RLock()
        self._storage_dir.mkdir(parents=True, exist_ok=True)

    def store(self, trace: MemoryTrace) -> None:
        """
        Store a memory trace to a JSON file.

        Args:
            trace: MemoryTrace to store
        """
        with self._lock:
            file_path = self._get_file_path(trace.request_id)
            trace_dict = trace.to_dict()

            with open(file_path, "w", encoding="utf-8") as f:
                json.dump(trace_dict, f, indent=2, ensure_ascii=False)

    def retrieve(self, request_id: uuid.UUID) -> MemoryTrace | None:
        """
        Retrieve a memory trace by request ID from file.

        Args:
            request_id: UUID of the trace to retrieve

        Returns:
            MemoryTrace if found, None otherwise
        """
        with self._lock:
            file_path = self._get_file_path(request_id)

            if not file_path.exists():
                return None

            try:
                with open(file_path, encoding="utf-8") as f:
                    data = json.load(f)
                return MemoryTrace.from_dict(data)
            except (json.JSONDecodeError, KeyError, ValueError):
                # Handle corrupted or invalid files
                return None

    def list_all(self) -> list[MemoryTrace]:
        """
        List all stored memory traces from files.

        Returns:
            List of all MemoryTrace objects sorted by created_at
        """
        with self._lock:
            traces: list[MemoryTrace] = []

            for file_path in self._storage_dir.glob("*.json"):
                try:
                    with open(file_path, encoding="utf-8") as f:
                        data = json.load(f)
                    trace = MemoryTrace.from_dict(data)
                    traces.append(trace)
                except (json.JSONDecodeError, KeyError, ValueError):
                    # Skip corrupted files
                    continue

            return sorted(traces, key=lambda t: t.created_at)

    def clear(self) -> None:
        """Clear all stored trace files."""
        with self._lock:
            for file_path in self._storage_dir.glob("*.json"):
                file_path.unlink()

    def _get_file_path(self, request_id: uuid.UUID) -> Path:
        """
        Get the file path for a given request ID.

        Args:
            request_id: UUID to get file path for

        Returns:
            Path object for the trace file
        """
        return self._storage_dir / f"{request_id}.json"
