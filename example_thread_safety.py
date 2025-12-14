"""
Quick demonstration of thread-safety in memory stores.
"""

import threading
import time
from pathlib import Path
from tempfile import TemporaryDirectory

from aris.input_interface import InputInterface
from aris.memory_store import FileBackedMemoryStore, InMemoryStore, MemoryTrace
from aris.reasoning_engine import ReasoningEngine


def test_concurrent_operations(store_name: str) -> None:
    """Test concurrent operations on a memory store."""
    print(f"\nTesting {store_name}...")

    if store_name == "InMemoryStore":
        store = InMemoryStore()
    else:
        tmpdir = TemporaryDirectory()
        store = FileBackedMemoryStore(Path(tmpdir.name))

    input_interface = InputInterface()
    reasoning_engine = ReasoningEngine()

    def worker(thread_id: int) -> None:
        """Worker function that stores multiple traces."""
        for i in range(5):
            packet = input_interface.process_input(f"Thread {thread_id}, message {i}")
            result = reasoning_engine.reason(packet)
            trace = MemoryTrace.from_reasoning_result(result)
            store.store(trace)
            time.sleep(0.001)  # Small delay to increase contention

    # Create and start 10 threads
    threads = [threading.Thread(target=worker, args=(i,)) for i in range(10)]

    start_time = time.time()
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    elapsed = time.time() - start_time

    # Verify all traces were stored
    traces = store.list_all()
    expected = 10 * 5  # 10 threads * 5 messages each

    print("  Threads: 10")
    print("  Messages per thread: 5")
    print(f"  Expected total: {expected}")
    print(f"  Actual total: {len(traces)}")
    print(f"  Time elapsed: {elapsed:.3f}s")
    print(f"  Result: {'✓ PASS' if len(traces) == expected else '✗ FAIL'}")


def main() -> None:
    """Run thread-safety tests."""
    print("=" * 70)
    print("Thread-Safety Test")
    print("=" * 70)

    test_concurrent_operations("InMemoryStore")
    test_concurrent_operations("FileBackedMemoryStore")

    print("\n" + "=" * 70)


if __name__ == "__main__":
    main()
