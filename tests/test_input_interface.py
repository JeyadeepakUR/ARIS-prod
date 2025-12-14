"""
Unit tests for the InputInterface and InputPacket.

Tests verify proper input processing, metadata generation, and type safety.
"""

import uuid
from datetime import UTC, datetime

import pytest

from aris.input_interface import InputInterface, InputPacket


class TestInputPacket:
    """Tests for the InputPacket dataclass."""

    def test_input_packet_creation(self) -> None:
        """Test that InputPacket can be created with required fields."""
        text = "test input"
        request_id = uuid.uuid4()
        timestamp = datetime.now(UTC)

        packet = InputPacket(text=text, request_id=request_id, timestamp=timestamp)

        assert packet.text == text
        assert packet.request_id == request_id
        assert packet.timestamp == timestamp

    def test_input_packet_immutable(self) -> None:
        """Test that InputPacket is immutable (frozen)."""
        packet = InputPacket(
            text="test", request_id=uuid.uuid4(), timestamp=datetime.now(UTC)
        )

        with pytest.raises(AttributeError):
            packet.text = "modified"  # type: ignore[misc]

    def test_input_packet_types(self) -> None:
        """Test that InputPacket enforces correct types."""
        request_id = uuid.uuid4()
        timestamp = datetime.now(UTC)

        packet = InputPacket(text="test", request_id=request_id, timestamp=timestamp)

        assert isinstance(packet.text, str)
        assert isinstance(packet.request_id, uuid.UUID)
        assert isinstance(packet.timestamp, datetime)


class TestInputInterface:
    """Tests for the InputInterface class."""

    def test_process_input_basic(self) -> None:
        """Test basic input processing."""
        interface = InputInterface()
        raw_text = "Hello, world!"

        packet = interface.process_input(raw_text)

        assert packet.text == "Hello, world!"
        assert isinstance(packet.request_id, uuid.UUID)
        assert isinstance(packet.timestamp, datetime)

    def test_process_input_whitespace_trimming(self) -> None:
        """Test that leading/trailing whitespace is trimmed."""
        interface = InputInterface()
        raw_text = "  \n\t  Hello  \t\n  "

        packet = interface.process_input(raw_text)

        assert packet.text == "Hello"

    def test_process_input_empty_string(self) -> None:
        """Test processing of empty string."""
        interface = InputInterface()

        packet = interface.process_input("")

        assert packet.text == ""
        assert isinstance(packet.request_id, uuid.UUID)
        assert isinstance(packet.timestamp, datetime)

    def test_process_input_whitespace_only(self) -> None:
        """Test processing of whitespace-only input."""
        interface = InputInterface()

        packet = interface.process_input("   \n\t   ")

        assert packet.text == ""

    def test_process_input_preserves_internal_whitespace(self) -> None:
        """Test that internal whitespace is preserved."""
        interface = InputInterface()
        raw_text = "  Hello    world  "

        packet = interface.process_input(raw_text)

        assert packet.text == "Hello    world"

    def test_process_input_unique_request_ids(self) -> None:
        """Test that each call generates a unique request ID."""
        interface = InputInterface()

        packet1 = interface.process_input("test 1")
        packet2 = interface.process_input("test 2")

        assert packet1.request_id != packet2.request_id

    def test_process_input_utc_timestamp(self) -> None:
        """Test that timestamp is in UTC."""
        interface = InputInterface()

        before = datetime.now(UTC)
        packet = interface.process_input("test")
        after = datetime.now(UTC)

        assert before <= packet.timestamp <= after
        assert packet.timestamp.tzinfo is not None
        assert packet.timestamp.tzinfo.tzname(None) == "UTC"

    def test_process_input_multiline_text(self) -> None:
        """Test processing of multiline text."""
        interface = InputInterface()
        raw_text = "Line 1\nLine 2\nLine 3"

        packet = interface.process_input(raw_text)

        assert packet.text == "Line 1\nLine 2\nLine 3"

    def test_process_input_special_characters(self) -> None:
        """Test processing of text with special characters."""
        interface = InputInterface()
        raw_text = "Hello! @#$%^&*() 你好 🚀"

        packet = interface.process_input(raw_text)

        assert packet.text == raw_text

    def test_process_input_returns_input_packet(self) -> None:
        """Test that process_input returns an InputPacket instance."""
        interface = InputInterface()

        packet = interface.process_input("test")

        assert isinstance(packet, InputPacket)
