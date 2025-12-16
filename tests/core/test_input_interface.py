"""
Unit tests for the InputInterface and InputPacket (Module 1).

Validates normalization, validation constraints, and metadata safety.
"""

import uuid
from datetime import UTC, datetime

import pytest

from aris.core.input_interface import (
    EmptyInputError,
    InputInterface,
    InputPacket,
    MaxLengthExceededError,
    MissingSourceError,
)


class TestInputPacket:
    """Tests for the InputPacket dataclass."""

    def test_input_packet_creation(self) -> None:
        request_id = uuid.uuid4()
        timestamp = datetime.now(UTC)

        packet = InputPacket(
            text="test input",
            source="cli",
            request_id=request_id,
            timestamp=timestamp,
        )

        assert packet.text == "test input"
        assert packet.source == "cli"
        assert packet.request_id == request_id
        assert packet.timestamp == timestamp

    def test_input_packet_immutable(self) -> None:
        packet = InputPacket(
            text="test",
            source="unit-test",
            request_id=uuid.uuid4(),
            timestamp=datetime.now(UTC),
        )

        with pytest.raises(AttributeError):
            packet.text = "modified"  # type: ignore[misc]


class TestInputInterface:
    """Tests for the InputInterface class."""

    def test_accept_basic(self) -> None:
        interface = InputInterface()

        packet = interface.accept("Hello, world!", source="cli")

        assert packet.text == "Hello, world!"
        assert packet.source == "cli"
        assert isinstance(packet.request_id, uuid.UUID)
        assert isinstance(packet.timestamp, datetime)

    def test_accept_trims_and_normalizes_newlines(self) -> None:
        interface = InputInterface()
        raw_text = "\r\n  Hello\nWorld  \r"

        packet = interface.accept(raw_text, source="cli")

        assert packet.text == "Hello\nWorld"

    def test_accept_rejects_empty(self) -> None:
        interface = InputInterface()

        with pytest.raises(EmptyInputError):
            interface.accept("   \n\t   ", source="cli")

    def test_accept_enforces_max_length(self) -> None:
        interface = InputInterface(max_length=5)

        with pytest.raises(MaxLengthExceededError) as exc:
            interface.accept("abcdef", source="cli")

        assert exc.value.max_length == 5
        assert exc.value.actual_length == 6

    def test_accept_requires_source(self) -> None:
        interface = InputInterface()

        with pytest.raises(MissingSourceError):
            interface.accept("hello", source="   ")

    def test_accept_preserves_internal_whitespace(self) -> None:
        interface = InputInterface()
        packet = interface.accept("  Hello    world  ", source="cli")

        assert packet.text == "Hello    world"

    def test_accept_unique_request_ids(self) -> None:
        interface = InputInterface()

        packet1 = interface.accept("test 1", source="cli")
        packet2 = interface.accept("test 2", source="cli")

        assert packet1.request_id != packet2.request_id

    def test_accept_utc_timestamp(self) -> None:
        interface = InputInterface()

        before = datetime.now(UTC)
        packet = interface.accept("test", source="cli")
        after = datetime.now(UTC)

        assert before <= packet.timestamp <= after
        assert packet.timestamp.tzinfo is not None
        assert packet.timestamp.tzinfo.tzname(None) == "UTC"

    def test_accept_multiline_text(self) -> None:
        interface = InputInterface()
        raw_text = "Line 1\r\nLine 2\rLine 3"

        packet = interface.accept(raw_text, source="cli")

        assert packet.text == "Line 1\nLine 2\nLine 3"

    def test_accept_returns_input_packet(self) -> None:
        interface = InputInterface()

        packet = interface.accept("test", source="cli")

        assert isinstance(packet, InputPacket)
