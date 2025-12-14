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
    packet1 = interface.accept("Hello, ARIS!", source="example")
    print(f"  Text: {packet1.text}")
    print(f"  Request ID: {packet1.request_id}")
    print(f"  Timestamp: {packet1.timestamp}")
    print()

    # Example 2: Input with whitespace
    print("Example 2: Input with whitespace trimming")
    packet2 = interface.accept("  \n  Trimmed input  \t  ", source="example")
    print(f"  Text: '{packet2.text}'")
    print(f"  Request ID: {packet2.request_id}")
    print(f"  Timestamp: {packet2.timestamp}")
    print()

    # Example 3: Empty input
    print("Example 3: Empty input")
    # Example 4: Multiline input
    print("Example 4: Multiline input")
    packet4 = interface.accept("Line 1\r\nLine 2\rLine 3", source="example")
    print(f"  Text: '{packet4.text}'")
    print(f"  Request ID: {packet4.request_id}")
    print(f"  Timestamp: {packet4.timestamp}")


if __name__ == "__main__":
    main()
