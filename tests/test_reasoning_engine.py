"""
Unit tests for the ReasoningEngine and ReasoningResult.

Tests verify proper reasoning generation, confidence scoring, and API stability.
"""

import uuid
from datetime import UTC, datetime

import pytest

from aris.input_interface import InputPacket
from aris.reasoning_engine import ReasoningEngine, ReasoningResult


class TestReasoningResult:
    """Tests for the ReasoningResult dataclass."""

    def test_reasoning_result_creation(self) -> None:
        """Test that ReasoningResult can be created with required fields."""
        input_packet = InputPacket(
            text="test", request_id=uuid.uuid4(), timestamp=datetime.now(UTC)
        )
        steps = ["Step 1", "Step 2", "Step 3"]
        confidence = 0.85

        result = ReasoningResult(
            reasoning_steps=steps, confidence_score=confidence, input_packet=input_packet
        )

        assert result.reasoning_steps == steps
        assert result.confidence_score == confidence
        assert result.input_packet == input_packet

    def test_reasoning_result_immutable(self) -> None:
        """Test that ReasoningResult is immutable (frozen)."""
        input_packet = InputPacket(
            text="test", request_id=uuid.uuid4(), timestamp=datetime.now(UTC)
        )
        result = ReasoningResult(
            reasoning_steps=["Step 1"],
            confidence_score=0.5,
            input_packet=input_packet,
        )

        with pytest.raises(AttributeError):
            result.confidence_score = 0.9  # type: ignore[misc]

    def test_reasoning_result_confidence_validation_low(self) -> None:
        """Test that confidence score below 0.0 raises ValueError."""
        input_packet = InputPacket(
            text="test", request_id=uuid.uuid4(), timestamp=datetime.now(UTC)
        )

        with pytest.raises(ValueError, match="must be between 0.0 and 1.0"):
            ReasoningResult(
                reasoning_steps=["Step 1"],
                confidence_score=-0.1,
                input_packet=input_packet,
            )

    def test_reasoning_result_confidence_validation_high(self) -> None:
        """Test that confidence score above 1.0 raises ValueError."""
        input_packet = InputPacket(
            text="test", request_id=uuid.uuid4(), timestamp=datetime.now(UTC)
        )

        with pytest.raises(ValueError, match="must be between 0.0 and 1.0"):
            ReasoningResult(
                reasoning_steps=["Step 1"],
                confidence_score=1.1,
                input_packet=input_packet,
            )

    def test_reasoning_result_confidence_edge_cases(self) -> None:
        """Test that confidence score at 0.0 and 1.0 are valid."""
        input_packet = InputPacket(
            text="test", request_id=uuid.uuid4(), timestamp=datetime.now(UTC)
        )

        # Test 0.0
        result_min = ReasoningResult(
            reasoning_steps=["Step 1"],
            confidence_score=0.0,
            input_packet=input_packet,
        )
        assert result_min.confidence_score == 0.0

        # Test 1.0
        result_max = ReasoningResult(
            reasoning_steps=["Step 1"],
            confidence_score=1.0,
            input_packet=input_packet,
        )
        assert result_max.confidence_score == 1.0


class TestReasoningEngine:
    """Tests for the ReasoningEngine class."""

    def test_reason_basic(self) -> None:
        """Test basic reasoning functionality."""
        engine = ReasoningEngine()
        input_packet = InputPacket(
            text="Hello world", request_id=uuid.uuid4(), timestamp=datetime.now(UTC)
        )

        result = engine.reason(input_packet)

        assert isinstance(result, ReasoningResult)
        assert isinstance(result.reasoning_steps, list)
        assert len(result.reasoning_steps) >= 3
        assert len(result.reasoning_steps) <= 5
        assert 0.0 <= result.confidence_score <= 1.0
        assert result.input_packet == input_packet

    def test_reason_steps_are_strings(self) -> None:
        """Test that all reasoning steps are strings."""
        engine = ReasoningEngine()
        input_packet = InputPacket(
            text="test input", request_id=uuid.uuid4(), timestamp=datetime.now(UTC)
        )

        result = engine.reason(input_packet)

        for step in result.reasoning_steps:
            assert isinstance(step, str)
            assert len(step) > 0

    def test_reason_empty_input(self) -> None:
        """Test reasoning with empty input."""
        engine = ReasoningEngine()
        input_packet = InputPacket(
            text="", request_id=uuid.uuid4(), timestamp=datetime.now(UTC)
        )

        result = engine.reason(input_packet)

        assert len(result.reasoning_steps) >= 3
        assert len(result.reasoning_steps) <= 5
        assert 0.0 <= result.confidence_score <= 1.0

    def test_reason_single_word(self) -> None:
        """Test reasoning with single word input."""
        engine = ReasoningEngine()
        input_packet = InputPacket(
            text="hello", request_id=uuid.uuid4(), timestamp=datetime.now(UTC)
        )

        result = engine.reason(input_packet)

        assert len(result.reasoning_steps) >= 3
        assert any("Single-word" in step or "word" in step for step in result.reasoning_steps)

    def test_reason_multi_word(self) -> None:
        """Test reasoning with multi-word input."""
        engine = ReasoningEngine()
        input_packet = InputPacket(
            text="This is a longer test input with many words",
            request_id=uuid.uuid4(),
            timestamp=datetime.now(UTC),
        )

        result = engine.reason(input_packet)

        assert len(result.reasoning_steps) >= 3
        assert len(result.reasoning_steps) <= 5

    def test_confidence_score_deterministic(self) -> None:
        """Test that confidence scores are deterministic for same input."""
        engine = ReasoningEngine()
        text = "Test input for consistency"

        # Create two different input packets with same text
        packet1 = InputPacket(
            text=text, request_id=uuid.uuid4(), timestamp=datetime.now(UTC)
        )
        packet2 = InputPacket(
            text=text, request_id=uuid.uuid4(), timestamp=datetime.now(UTC)
        )

        result1 = engine.reason(packet1)
        result2 = engine.reason(packet2)

        # Confidence should be same for same text length
        assert result1.confidence_score == result2.confidence_score

    def test_confidence_score_empty_input(self) -> None:
        """Test that empty input has low confidence."""
        engine = ReasoningEngine()
        input_packet = InputPacket(
            text="", request_id=uuid.uuid4(), timestamp=datetime.now(UTC)
        )

        result = engine.reason(input_packet)

        assert result.confidence_score < 0.5

    def test_confidence_score_short_input(self) -> None:
        """Test that short input has moderate confidence."""
        engine = ReasoningEngine()
        input_packet = InputPacket(
            text="Hi", request_id=uuid.uuid4(), timestamp=datetime.now(UTC)
        )

        result = engine.reason(input_packet)

        assert 0.4 <= result.confidence_score <= 0.7

    def test_confidence_score_long_input(self) -> None:
        """Test that long input has high confidence."""
        engine = ReasoningEngine()
        long_text = (
            "This is a much longer input string that contains many characters and "
            "should result in a higher confidence score from the reasoning engine."
        )
        input_packet = InputPacket(
            text=long_text, request_id=uuid.uuid4(), timestamp=datetime.now(UTC)
        )

        result = engine.reason(input_packet)

        assert result.confidence_score > 0.7

    def test_reason_includes_request_id(self) -> None:
        """Test that reasoning steps include request ID."""
        engine = ReasoningEngine()
        request_id = uuid.uuid4()
        input_packet = InputPacket(
            text="test", request_id=request_id, timestamp=datetime.now(UTC)
        )

        result = engine.reason(input_packet)

        # Check that request ID appears in reasoning steps
        assert any(str(request_id) in step for step in result.reasoning_steps)

    def test_reason_includes_timestamp(self) -> None:
        """Test that reasoning steps include timestamp information."""
        engine = ReasoningEngine()
        timestamp = datetime.now(UTC)
        input_packet = InputPacket(
            text="test", request_id=uuid.uuid4(), timestamp=timestamp
        )

        result = engine.reason(input_packet)

        # Check that timestamp info appears in reasoning steps
        assert any("timestamp" in step.lower() for step in result.reasoning_steps)

    def test_reason_includes_input_characteristics(self) -> None:
        """Test that reasoning steps include input text characteristics."""
        engine = ReasoningEngine()
        input_packet = InputPacket(
            text="Hello world test", request_id=uuid.uuid4(), timestamp=datetime.now(UTC)
        )

        result = engine.reason(input_packet)

        # Check that character/word info appears
        steps_text = " ".join(result.reasoning_steps).lower()
        assert "character" in steps_text or "word" in steps_text

    def test_reason_api_stability(self) -> None:
        """Test that the API returns expected types for future LLM compatibility."""
        engine = ReasoningEngine()
        input_packet = InputPacket(
            text="API stability test", request_id=uuid.uuid4(), timestamp=datetime.now(UTC)
        )

        # This is the stable API contract
        result: ReasoningResult = engine.reason(input_packet)

        # Verify return type structure that future LLMs must maintain
        assert hasattr(result, "reasoning_steps")
        assert hasattr(result, "confidence_score")
        assert hasattr(result, "input_packet")
        assert isinstance(result.reasoning_steps, list)
        assert isinstance(result.confidence_score, float)
        assert isinstance(result.input_packet, InputPacket)
