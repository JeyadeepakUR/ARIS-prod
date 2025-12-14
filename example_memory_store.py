"""
Example usage of MemoryStore with InputInterface and ReasoningEngine.

Demonstrates the full pipeline: input processing, reasoning, and memory storage.
"""

import uuid
from pathlib import Path
from tempfile import TemporaryDirectory

from aris.input_interface import InputInterface
from aris.memory_store import (
    FileBackedMemoryStore,
    InMemoryStore,
    MemoryStore,
    MemoryTrace,
)
from aris.reasoning_engine import ReasoningEngine


def demonstrate_store(store: MemoryStore, store_name: str) -> None:
    """
    Demonstrate memory store operations.

    Args:
        store: MemoryStore implementation to demonstrate
        store_name: Name of the store for display purposes
    """
    print("=" * 70)
    print(f"{store_name}")
    print("=" * 70)

    input_interface = InputInterface()
    reasoning_engine = ReasoningEngine()

    # Process multiple inputs
    inputs = [
        "Hello, ARIS!",
        "What is the meaning of life?",
        "Explain quantum computing",
    ]

    print(f"\nStoring {len(inputs)} reasoning traces...\n")

    request_ids: list[uuid.UUID] = []

    for text in inputs:
        # Full pipeline: input -> reasoning -> storage
        packet = input_interface.accept(text, source="example")
        result = reasoning_engine.reason(packet)
        trace = MemoryTrace.from_reasoning_result(result)

        store.store(trace)
        request_ids.append(packet.request_id)

        print(f"Stored: '{text}'")
        print(f"  Request ID: {packet.request_id}")
        print(f"  Confidence: {result.confidence_score:.2f}")
        print()

    # Retrieve by request_id
    print("Retrieving first trace by request ID...")
    retrieved = store.retrieve(request_ids[0])
    if retrieved:
        print(f"  Retrieved: {retrieved.input_data['text']}")
        print(f"  Steps: {len(retrieved.reasoning_data['reasoning_steps'])} steps")
        print()

    # List all traces
    print("Listing all traces...")
    all_traces = store.list_all()
    print(f"  Total traces: {len(all_traces)}")
    for i, trace in enumerate(all_traces, 1):
        confidence = trace.reasoning_data["confidence_score"]
        print(
            f"  {i}. {trace.input_data['text']} (confidence: {confidence})"
        )
    print()


def demonstrate_json_format() -> None:
    """Demonstrate the JSON file format."""
    print("=" * 70)
    print("JSON File Format Demonstration")
    print("=" * 70)

    with TemporaryDirectory() as tmpdir:
        store = FileBackedMemoryStore(Path(tmpdir))

        # Create and store a trace
        input_interface = InputInterface()
        reasoning_engine = ReasoningEngine()

        packet = input_interface.accept("Sample input for JSON demo", source="example")
        result = reasoning_engine.reason(packet)
        trace = MemoryTrace.from_reasoning_result(result)

        store.store(trace)

        # Read and display the JSON file
        json_file = Path(tmpdir) / f"{packet.request_id}.json"
        print(f"\nJSON file: {json_file.name}")
        print("\nContents:")
        print("-" * 70)
        with open(json_file, encoding="utf-8") as f:
            print(f.read())
        print("-" * 70)


def demonstrate_persistence() -> None:
    """Demonstrate persistence across store instances."""
    print("\n" + "=" * 70)
    print("Persistence Demonstration")
    print("=" * 70)

    with TemporaryDirectory() as tmpdir:
        storage_path = Path(tmpdir)

        # Store with first instance
        print("\nCreating first store instance and storing data...")
        store1 = FileBackedMemoryStore(storage_path)

        input_interface = InputInterface()
        reasoning_engine = ReasoningEngine()

        packet = input_interface.accept("Persistent data test", source="example")
        result = reasoning_engine.reason(packet)
        trace = MemoryTrace.from_reasoning_result(result)

        store1.store(trace)
        print(f"Stored trace with request_id: {packet.request_id}")

        # Retrieve with second instance
        print("\nCreating second store instance and retrieving data...")
        store2 = FileBackedMemoryStore(storage_path)

        retrieved = store2.retrieve(packet.request_id)
        if retrieved:
            print(f"✓ Successfully retrieved: '{retrieved.input_data['text']}'")
            print("  Data persisted across store instances!")
        else:
            print("✗ Failed to retrieve data")


def main() -> None:
    """Run all demonstrations."""
    print("\n" + "★" * 70)
    print("ARIS Memory Store Demonstration")
    print("★" * 70 + "\n")

    # Demonstrate in-memory store
    in_memory_store = InMemoryStore()
    demonstrate_store(in_memory_store, "In-Memory Store")

    # Demonstrate file-backed store
    with TemporaryDirectory() as tmpdir:
        file_store = FileBackedMemoryStore(Path(tmpdir))
        demonstrate_store(file_store, "File-Backed Store")

    # Demonstrate JSON format
    demonstrate_json_format()

    # Demonstrate persistence
    demonstrate_persistence()

    print("\n" + "★" * 70)
    print("Demonstration Complete")
    print("★" * 70 + "\n")


if __name__ == "__main__":
    main()
