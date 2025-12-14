"""
Example usage of ReasoningEngine with InputInterface.

Demonstrates the integration of InputInterface and ReasoningEngine to process
user input and generate reasoning steps.
"""

from aris.input_interface import InputInterface
from aris.reasoning_engine import ReasoningEngine


def main() -> None:
    """Run example usage of ReasoningEngine."""
    input_interface = InputInterface()
    reasoning_engine = ReasoningEngine()

    # Example 1: Simple query
    print("=" * 70)
    print("Example 1: Simple query")
    print("=" * 70)
    packet1 = input_interface.process_input("Hello")
    result1 = reasoning_engine.reason(packet1)
    print(f"Input: '{result1.input_packet.text}'")
    print(f"Confidence: {result1.confidence_score:.2f}")
    print("Reasoning steps:")
    for step in result1.reasoning_steps:
        print(f"  {step}")
    print()

    # Example 2: Empty input
    print("=" * 70)
    print("Example 2: Empty input")
    print("=" * 70)
    packet2 = input_interface.process_input("")
    result2 = reasoning_engine.reason(packet2)
    print(f"Input: '{result2.input_packet.text}'")
    print(f"Confidence: {result2.confidence_score:.2f}")
    print("Reasoning steps:")
    for step in result2.reasoning_steps:
        print(f"  {step}")
    print()

    # Example 3: Short phrase
    print("=" * 70)
    print("Example 3: Short phrase")
    print("=" * 70)
    packet3 = input_interface.process_input("What is AI?")
    result3 = reasoning_engine.reason(packet3)
    print(f"Input: '{result3.input_packet.text}'")
    print(f"Confidence: {result3.confidence_score:.2f}")
    print("Reasoning steps:")
    for step in result3.reasoning_steps:
        print(f"  {step}")
    print()

    # Example 4: Longer detailed input
    print("=" * 70)
    print("Example 4: Longer detailed input")
    print("=" * 70)
    packet4 = input_interface.process_input(
        "Can you explain how the reasoning engine works and what confidence scores mean?"
    )
    result4 = reasoning_engine.reason(packet4)
    print(f"Input: '{result4.input_packet.text}'")
    print(f"Confidence: {result4.confidence_score:.2f}")
    print("Reasoning steps:")
    for step in result4.reasoning_steps:
        print(f"  {step}")
    print()

    # Example 5: Multiline input
    print("=" * 70)
    print("Example 5: Multiline input")
    print("=" * 70)
    multiline_text = """Generate a detailed plan:
1. First step
2. Second step
3. Third step"""
    packet5 = input_interface.process_input(multiline_text)
    result5 = reasoning_engine.reason(packet5)
    print(f"Input (first 50 chars): '{result5.input_packet.text[:50]}...'")
    print(f"Confidence: {result5.confidence_score:.2f}")
    print("Reasoning steps:")
    for step in result5.reasoning_steps:
        print(f"  {step}")


if __name__ == "__main__":
    main()
