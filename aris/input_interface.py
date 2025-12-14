"""
Input interface for processing user text input with request tracking.

This module provides a clean interface for accepting user input and packaging it
with metadata (request ID and timestamp) for downstream processing.
"""

import uuid
from dataclasses import dataclass
from datetime import UTC, datetime


@dataclass(frozen=True)
class InputPacket:
    """
    Immutable data packet containing user input with tracking metadata.

    Attributes:
        text: User input text (whitespace-trimmed)
        request_id: Unique UUIDv4 identifier for this request
        timestamp: UTC timestamp when the packet was created
    """

    text: str
    request_id: uuid.UUID
    timestamp: datetime


class InputInterface:
    """
    Interface for processing raw user text input into structured packets.

    This class handles the conversion of raw user text into InputPacket objects
    with automatically generated request IDs and timestamps. No AI processing or
    complex preprocessing is performed - only whitespace trimming.
    """

    def process_input(self, raw_text: str) -> InputPacket:
        """
        Process raw user text into a structured InputPacket.

        Args:
            raw_text: Raw text input from the user

        Returns:
            InputPacket containing trimmed text, unique request ID, and UTC timestamp
        """
        trimmed_text = raw_text.strip()
        request_id = uuid.uuid4()
        timestamp = datetime.now(UTC)

        return InputPacket(text=trimmed_text, request_id=request_id, timestamp=timestamp)
