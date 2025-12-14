"""
Unit tests for MemoryStore implementations.

Tests verify proper storage, retrieval, thread-safety, and persistence.
"""

import json
import threading
import time
import uuid
from datetime import UTC, datetime
from pathlib import Path
from tempfile import TemporaryDirectory

import pytest

from aris.input_interface import InputInterface
from aris.memory_store import (
    FileBackedMemoryStore,
    InMemoryStore,
    MemoryStore,
    MemoryTrace,
)
from aris.reasoning_engine import ReasoningEngine


class TestMemoryTrace:
    """Tests for the MemoryTrace dataclass."""

    def test_memory_trace_creation(self) -> None:
        """Test that MemoryTrace can be created with required fields."""
        request_id = uuid.uuid4()
        input_data = {"text": "test", "request_id": str(request_id), "timestamp": "2024-01-01"}
        reasoning_data = {"reasoning_steps": ["Step 1"], "confidence_score": 0.8}
        created_at = datetime.now(UTC)

        trace = MemoryTrace(
            request_id=request_id,
            input_data=input_data,
            reasoning_data=reasoning_data,
            created_at=created_at,
        )

        assert trace.request_id == request_id
        assert trace.input_data == input_data
        assert trace.reasoning_data == reasoning_data
        assert trace.created_at == created_at

    def test_memory_trace_immutable(self) -> None:
        """Test that MemoryTrace is immutable (frozen)."""
        trace = MemoryTrace(
            request_id=uuid.uuid4(),
            input_data={"text": "test"},
            reasoning_data={"steps": []},
            created_at=datetime.now(UTC),
        )

        with pytest.raises(AttributeError):
            trace.request_id = uuid.uuid4()  # type: ignore[misc]

    def test_memory_trace_from_reasoning_result(self) -> None:
        """Test creating MemoryTrace from ReasoningResult."""
        interface = InputInterface()
        engine = ReasoningEngine()

        packet = interface.accept("Test input", source="test")
        result = engine.reason(packet)

        trace = MemoryTrace.from_reasoning_result(result)

        assert trace.request_id == packet.request_id
        assert trace.input_data["text"] == "Test input"
        assert "reasoning_steps" in trace.reasoning_data
        assert "confidence_score" in trace.reasoning_data

    def test_memory_trace_to_dict(self) -> None:
        """Test converting MemoryTrace to dictionary."""
        request_id = uuid.uuid4()
        trace = MemoryTrace(
            request_id=request_id,
            input_data={"text": "test"},
            reasoning_data={"steps": ["Step 1"]},
            created_at=datetime.now(UTC),
        )

        trace_dict = trace.to_dict()

        assert trace_dict["request_id"] == str(request_id)
        assert trace_dict["input"] == {"text": "test"}
        assert trace_dict["reasoning"] == {"steps": ["Step 1"]}
        assert "created_at" in trace_dict

    def test_memory_trace_from_dict(self) -> None:
        """Test creating MemoryTrace from dictionary."""
        request_id = uuid.uuid4()
        created_at = datetime.now(UTC)
        data: dict[str, object] = {
            "request_id": str(request_id),
            "input": {"text": "test"},
            "reasoning": {"steps": ["Step 1"]},
            "created_at": created_at.isoformat(),
        }

        trace = MemoryTrace.from_dict(data)

        assert trace.request_id == request_id
        assert trace.input_data == {"text": "test"}
        assert trace.reasoning_data == {"steps": ["Step 1"]}
        assert trace.created_at == created_at

    def test_memory_trace_round_trip(self) -> None:
        """Test that MemoryTrace can be converted to dict and back."""
        original = MemoryTrace(
            request_id=uuid.uuid4(),
            input_data={"text": "test", "extra": "data"},
            reasoning_data={"steps": ["Step 1", "Step 2"], "score": 0.9},
            evaluation_data={"overall_score": 0.8},
            created_at=datetime.now(UTC),
        )

        trace_dict = original.to_dict()
        restored = MemoryTrace.from_dict(trace_dict)

        assert restored.request_id == original.request_id
        assert restored.input_data == original.input_data
        assert restored.reasoning_data == original.reasoning_data
        assert restored.evaluation_data == original.evaluation_data
        assert restored.created_at == original.created_at

    def test_memory_trace_includes_optional_evaluation(self) -> None:
        interface = InputInterface()
        engine = ReasoningEngine()

        packet = interface.accept("Eval input", source="test")
        result = engine.reason(packet)

        evaluation_data = {
            "step_count_score": 0.9,
            "coherence_score": 1.0,
            "confidence_alignment_score": 0.8,
            "input_coverage_score": 0.7,
            "overall_score": 0.85,
        }

        trace = MemoryTrace.from_reasoning_result(result, evaluation=None)
        assert trace.evaluation_data is None

        trace_with_eval = MemoryTrace(
            request_id=packet.request_id,
            input_data=trace.input_data,
            reasoning_data=trace.reasoning_data,
            evaluation_data=evaluation_data,
            created_at=trace.created_at,
        )

        trace_dict = trace_with_eval.to_dict()
        assert "evaluation" in trace_dict
        restored = MemoryTrace.from_dict(trace_dict)
        assert restored.evaluation_data == evaluation_data


class TestInMemoryStore:
    """Tests for the InMemoryStore implementation."""

    def test_store_and_retrieve(self) -> None:
        """Test basic store and retrieve operations."""
        store = InMemoryStore()
        trace = MemoryTrace(
            request_id=uuid.uuid4(),
            input_data={"text": "test"},
            reasoning_data={"steps": ["Step 1"]},
            created_at=datetime.now(UTC),
        )

        store.store(trace)
        retrieved = store.retrieve(trace.request_id)

        assert retrieved is not None
        assert retrieved.request_id == trace.request_id

    def test_retrieve_nonexistent(self) -> None:
        """Test retrieving a trace that doesn't exist."""
        store = InMemoryStore()
        result = store.retrieve(uuid.uuid4())

        assert result is None

    def test_list_all_empty(self) -> None:
        """Test listing all traces when store is empty."""
        store = InMemoryStore()
        traces = store.list_all()

        assert traces == []

    def test_list_all_multiple(self) -> None:
        """Test listing all traces with multiple entries."""
        store = InMemoryStore()

        trace1 = MemoryTrace(
            request_id=uuid.uuid4(),
            input_data={"text": "test1"},
            reasoning_data={"steps": []},
            created_at=datetime.now(UTC),
        )
        time.sleep(0.01)  # Ensure different timestamps
        trace2 = MemoryTrace(
            request_id=uuid.uuid4(),
            input_data={"text": "test2"},
            reasoning_data={"steps": []},
            created_at=datetime.now(UTC),
        )

        store.store(trace1)
        store.store(trace2)

        traces = store.list_all()

        assert len(traces) == 2
        assert traces[0].created_at <= traces[1].created_at

    def test_clear(self) -> None:
        """Test clearing the store."""
        store = InMemoryStore()

        trace = MemoryTrace(
            request_id=uuid.uuid4(),
            input_data={"text": "test"},
            reasoning_data={"steps": []},
            created_at=datetime.now(UTC),
        )

        store.store(trace)
        assert len(store.list_all()) == 1

        store.clear()
        assert len(store.list_all()) == 0

    def test_store_overwrite(self) -> None:
        """Test that storing with same request_id overwrites."""
        store = InMemoryStore()
        request_id = uuid.uuid4()

        trace1 = MemoryTrace(
            request_id=request_id,
            input_data={"text": "original"},
            reasoning_data={"steps": []},
            created_at=datetime.now(UTC),
        )
        trace2 = MemoryTrace(
            request_id=request_id,
            input_data={"text": "updated"},
            reasoning_data={"steps": []},
            created_at=datetime.now(UTC),
        )

        store.store(trace1)
        store.store(trace2)

        retrieved = store.retrieve(request_id)
        assert retrieved is not None
        assert retrieved.input_data["text"] == "updated"

    def test_thread_safety_concurrent_stores(self) -> None:
        """Test thread-safety with concurrent store operations."""
        store = InMemoryStore()
        num_threads = 10
        traces_per_thread = 5

        def store_traces() -> None:
            for i in range(traces_per_thread):
                trace = MemoryTrace(
                    request_id=uuid.uuid4(),
                    input_data={"text": f"test_{i}"},
                    reasoning_data={"steps": []},
                    created_at=datetime.now(UTC),
                )
                store.store(trace)

        threads = [threading.Thread(target=store_traces) for _ in range(num_threads)]

        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()

        traces = store.list_all()
        assert len(traces) == num_threads * traces_per_thread

    def test_thread_safety_concurrent_read_write(self) -> None:
        """Test thread-safety with concurrent read and write operations."""
        store = InMemoryStore()
        request_id = uuid.uuid4()
        results: list[MemoryTrace | None] = []
        lock = threading.Lock()

        def write_trace() -> None:
            for i in range(10):
                trace = MemoryTrace(
                    request_id=request_id,
                    input_data={"text": f"version_{i}"},
                    reasoning_data={"steps": []},
                    created_at=datetime.now(UTC),
                )
                store.store(trace)
                time.sleep(0.001)

        def read_trace() -> None:
            for _ in range(10):
                result = store.retrieve(request_id)
                with lock:
                    results.append(result)
                time.sleep(0.001)

        writer = threading.Thread(target=write_trace)
        reader = threading.Thread(target=read_trace)

        writer.start()
        reader.start()
        writer.join()
        reader.join()

        # All reads should succeed (return trace or None)
        assert len(results) == 10


class TestFileBackedMemoryStore:
    """Tests for the FileBackedMemoryStore implementation."""

    def test_store_and_retrieve(self) -> None:
        """Test basic store and retrieve operations."""
        with TemporaryDirectory() as tmpdir:
            store = FileBackedMemoryStore(Path(tmpdir))
            trace = MemoryTrace(
                request_id=uuid.uuid4(),
                input_data={"text": "test"},
                reasoning_data={"steps": ["Step 1"]},
                created_at=datetime.now(UTC),
            )

            store.store(trace)
            retrieved = store.retrieve(trace.request_id)

            assert retrieved is not None
            assert retrieved.request_id == trace.request_id
            assert retrieved.input_data == trace.input_data

    def test_retrieve_nonexistent(self) -> None:
        """Test retrieving a trace that doesn't exist."""
        with TemporaryDirectory() as tmpdir:
            store = FileBackedMemoryStore(Path(tmpdir))
            result = store.retrieve(uuid.uuid4())

            assert result is None

    def test_persistence_across_instances(self) -> None:
        """Test that data persists across store instances."""
        with TemporaryDirectory() as tmpdir:
            storage_path = Path(tmpdir)
            request_id = uuid.uuid4()

            # Store with first instance
            store1 = FileBackedMemoryStore(storage_path)
            trace = MemoryTrace(
                request_id=request_id,
                input_data={"text": "persistent"},
                reasoning_data={"steps": []},
                created_at=datetime.now(UTC),
            )
            store1.store(trace)

            # Retrieve with second instance
            store2 = FileBackedMemoryStore(storage_path)
            retrieved = store2.retrieve(request_id)

            assert retrieved is not None
            assert retrieved.request_id == request_id
            assert retrieved.input_data["text"] == "persistent"

    def test_list_all_empty(self) -> None:
        """Test listing all traces when store is empty."""
        with TemporaryDirectory() as tmpdir:
            store = FileBackedMemoryStore(Path(tmpdir))
            traces = store.list_all()

            assert traces == []

    def test_list_all_multiple(self) -> None:
        """Test listing all traces with multiple entries."""
        with TemporaryDirectory() as tmpdir:
            store = FileBackedMemoryStore(Path(tmpdir))

            trace1 = MemoryTrace(
                request_id=uuid.uuid4(),
                input_data={"text": "test1"},
                reasoning_data={"steps": []},
                created_at=datetime.now(UTC),
            )
            time.sleep(0.01)
            trace2 = MemoryTrace(
                request_id=uuid.uuid4(),
                input_data={"text": "test2"},
                reasoning_data={"steps": []},
                created_at=datetime.now(UTC),
            )

            store.store(trace1)
            store.store(trace2)

            traces = store.list_all()

            assert len(traces) == 2
            assert traces[0].created_at <= traces[1].created_at

    def test_clear(self) -> None:
        """Test clearing the store."""
        with TemporaryDirectory() as tmpdir:
            store = FileBackedMemoryStore(Path(tmpdir))

            trace = MemoryTrace(
                request_id=uuid.uuid4(),
                input_data={"text": "test"},
                reasoning_data={"steps": []},
                created_at=datetime.now(UTC),
            )

            store.store(trace)
            assert len(store.list_all()) == 1

            store.clear()
            assert len(store.list_all()) == 0

    def test_json_format(self) -> None:
        """Test that stored files match the required JSON schema."""
        with TemporaryDirectory() as tmpdir:
            store = FileBackedMemoryStore(Path(tmpdir))
            request_id = uuid.uuid4()

            trace = MemoryTrace(
                request_id=request_id,
                input_data={"text": "test", "extra": "field"},
                reasoning_data={"steps": ["Step 1", "Step 2"], "confidence_score": 0.85},
                created_at=datetime.now(UTC),
            )

            store.store(trace)

            # Read the file directly and verify schema
            file_path = Path(tmpdir) / f"{request_id}.json"
            with open(file_path, encoding="utf-8") as f:
                data = json.load(f)

            assert "request_id" in data
            assert "input" in data
            assert "reasoning" in data
            assert "created_at" in data
            assert data["request_id"] == str(request_id)

    def test_corrupted_file_handling(self) -> None:
        """Test that corrupted JSON files are handled gracefully."""
        with TemporaryDirectory() as tmpdir:
            store = FileBackedMemoryStore(Path(tmpdir))
            request_id = uuid.uuid4()

            # Create a corrupted file
            file_path = Path(tmpdir) / f"{request_id}.json"
            with open(file_path, "w") as f:
                f.write("invalid json {")

            result = store.retrieve(request_id)
            assert result is None

            # list_all should skip corrupted files
            traces = store.list_all()
            assert len(traces) == 0

    def test_thread_safety_concurrent_stores(self) -> None:
        """Test thread-safety with concurrent store operations."""
        with TemporaryDirectory() as tmpdir:
            store = FileBackedMemoryStore(Path(tmpdir))
            num_threads = 5
            traces_per_thread = 3

            def store_traces() -> None:
                for i in range(traces_per_thread):
                    trace = MemoryTrace(
                        request_id=uuid.uuid4(),
                        input_data={"text": f"test_{i}"},
                        reasoning_data={"steps": []},
                        created_at=datetime.now(UTC),
                    )
                    store.store(trace)

            threads = [threading.Thread(target=store_traces) for _ in range(num_threads)]

            for thread in threads:
                thread.start()
            for thread in threads:
                thread.join()

            traces = store.list_all()
            assert len(traces) == num_threads * traces_per_thread

    def test_directory_creation(self) -> None:
        """Test that storage directory is created if it doesn't exist."""
        with TemporaryDirectory() as tmpdir:
            storage_path = Path(tmpdir) / "nested" / "storage"
            FileBackedMemoryStore(storage_path)

            assert storage_path.exists()
            assert storage_path.is_dir()


class TestMemoryStoreInterface:
    """Tests for the MemoryStore abstract interface."""

    def test_in_memory_implements_interface(self) -> None:
        """Test that InMemoryStore implements MemoryStore interface."""
        store: MemoryStore = InMemoryStore()

        assert hasattr(store, "store")
        assert hasattr(store, "retrieve")
        assert hasattr(store, "list_all")
        assert hasattr(store, "clear")

    def test_file_backed_implements_interface(self) -> None:
        """Test that FileBackedMemoryStore implements MemoryStore interface."""
        with TemporaryDirectory() as tmpdir:
            store: MemoryStore = FileBackedMemoryStore(Path(tmpdir))

            assert hasattr(store, "store")
            assert hasattr(store, "retrieve")
            assert hasattr(store, "list_all")
            assert hasattr(store, "clear")

    def test_polymorphic_usage(self) -> None:
        """Test that both implementations can be used polymorphically."""
        trace = MemoryTrace(
            request_id=uuid.uuid4(),
            input_data={"text": "test"},
            reasoning_data={"steps": []},
            created_at=datetime.now(UTC),
        )

        stores: list[MemoryStore] = [InMemoryStore()]

        with TemporaryDirectory() as tmpdir:
            stores.append(FileBackedMemoryStore(Path(tmpdir)))

            for store in stores:
                store.store(trace)
                retrieved = store.retrieve(trace.request_id)
                assert retrieved is not None
                assert retrieved.request_id == trace.request_id
