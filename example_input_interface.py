"""
Example usage of InputInterface.

Demonstrates basic usage of the InputInterface and InputPacket.
"""

from aris.input_interface import InputInterface


def main() -> None:
    """Run example usage of InputInterface."""
    interface = InputInterface()

    # Example 1: Basic input
    print("Example 1: Basic input")
    packet1 = interface.process_input("Hello, ARIS!")
    print(f"  Text: {packet1.text}")
    print(f"  Request ID: {packet1.request_id}")
    print(f"  Timestamp: {packet1.timestamp}")
    print()

    # Example 2: Input with whitespace
    print("Example 2: Input with whitespace trimming")
    packet2 = interface.process_input("  \n  Trimmed input  \t  ")
    print(f"  Text: '{packet2.text}'")
    print(f"  Request ID: {packet2.request_id}")
    print(f"  Timestamp: {packet2.timestamp}")
    print()

    # Example 3: Empty input
    print("Example 3: Empty input")
    packet3 = interface.process_input("")
    print(f"  Text: '{packet3.text}'")
    print(f"  Request ID: {packet3.request_id}")
    print(f"  Timestamp: {packet3.timestamp}")
    print()

    # Example 4: Multiline input
    print("Example 4: Multiline input")
    packet4 = interface.process_input("Line 1\nLine 2\nLine 3")
    print(f"  Text: '{packet4.text}'")
    print(f"  Request ID: {packet4.request_id}")
    print(f"  Timestamp: {packet4.timestamp}")


if __name__ == "__main__":
    main()
