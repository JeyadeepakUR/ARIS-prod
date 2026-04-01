"""
Reasoning engine for generating explicit reasoning steps.

This module provides a deterministic reasoning engine that generates structured
reasoning steps. The API is designed to be future-proof, allowing LLM-based
reasoning to replace the deterministic templates without refactoring callers.
"""

from dataclasses import dataclass

from aris.core.input_interface import InputPacket


@dataclass(frozen=True)
class ReasoningResult:
    """
    Immutable result from the reasoning engine.

    Attributes:
        reasoning_steps: Ordered list of explicit reasoning steps
        confidence_score: Confidence level for the reasoning (0.0 to 1.0)
        input_packet: Original input packet that was reasoned about
    """

    reasoning_steps: list[str]
    confidence_score: float
    input_packet: InputPacket

    def __post_init__(self) -> None:
        """Validate confidence score is in valid range."""
        if not 0.0 <= self.confidence_score <= 1.0:
            raise ValueError(
                f"Confidence score must be between 0.0 and 1.0, got {self.confidence_score}"
            )


class ReasoningEngine:
    """
    Engine for generating structured reasoning steps.

    This class provides a stable API for reasoning about user input. The current
    implementation uses deterministic templates, but the API is designed to support
    LLM-based reasoning in the future without requiring changes to calling code.

    Future implementations can replace the internal logic while maintaining the
    same public interface: reason(InputPacket) -> ReasoningResult.
    """

    def reason(self, input_packet: InputPacket) -> ReasoningResult:
        """
        Generate reasoning steps for the given input.

        Args:
            input_packet: Input packet containing user text and metadata

        Returns:
            ReasoningResult containing reasoning steps, confidence score, and input
        """
        reasoning_steps = self._generate_reasoning_steps(input_packet)
        confidence_score = self._compute_confidence_score(input_packet)

        return ReasoningResult(
            reasoning_steps=reasoning_steps,
            confidence_score=confidence_score,
            input_packet=input_packet,
        )

    def _generate_reasoning_steps(self, input_packet: InputPacket) -> list[str]:
        """
        Generate deterministic reasoning steps based on templates.

        This is a placeholder implementation using simple templates. Future versions
        can replace this with LLM-generated reasoning while maintaining the same
        return type.

        Args:
            input_packet: Input packet to reason about

        Returns:
            List of 3-5 reasoning steps as strings
        """
        text = input_packet.text
        text_length = len(text)
        word_count = len(text.split()) if text else 0

        steps: list[str] = [
            f"Step 1: Received input with request ID {input_packet.request_id}",
            f"Step 2: Input contains {text_length} characters and {word_count} words",
            f"Step 3: Input timestamp is {input_packet.timestamp.isoformat()}",
        ]

        # Add conditional steps based on input characteristics
        if not text:
            steps.append("Step 4: Input is empty, minimal processing required")
        elif word_count == 1:
            steps.append("Step 4: Single-word input detected, likely a simple command or query")
        elif word_count <= 5:
            steps.append("Step 5: Short input detected, processing as concise request")
        else:
            steps.append("Step 5: Multi-word input detected, processing as detailed request")

        # Ensure we return 3-5 steps
        return steps[:5]

    def _compute_confidence_score(self, input_packet: InputPacket) -> float:
        """
        Compute a deterministic confidence score.

        Uses simple heuristics based on input characteristics. Future implementations
        can replace this with LLM-based confidence estimation.

        Args:
            input_packet: Input packet to assess

        Returns:
            Confidence score between 0.0 and 1.0
        """
        text = input_packet.text
        text_length = len(text)

        # Empty input: low confidence
        if text_length == 0:
            return 0.2

        # Very short input (1-5 chars): moderate-low confidence
        if text_length <= 5:
            return 0.5

        # Short input (6-20 chars): moderate confidence
        if text_length <= 20:
            return 0.65

        # Medium input (21-100 chars): good confidence
        if text_length <= 100:
            return 0.8

        # Long input (100+ chars): high confidence
        return 0.9
