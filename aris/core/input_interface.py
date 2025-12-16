"""
Input interface for processing user text input with deterministic validation.

Module goals for safety-critical contexts:
- Explicit validation to reject empty or pathological input early.
- Deterministic normalization (newline canonicalization) to avoid surprises
  downstream and to keep logs comparable across platforms.
- Strict typing and immutability to prevent accidental mutation of metadata.
"""

import uuid
from dataclasses import dataclass
from datetime import UTC, datetime


class InputError(Exception):
    """Base exception for input handling errors."""


class InputValidationError(InputError):
    """Raised when user input fails validation constraints."""


class EmptyInputError(InputValidationError):
    """Raised when input is empty or whitespace-only to avoid null work items."""


class MaxLengthExceededError(InputValidationError):
    """Raised when input exceeds the configured maximum length to bound resources."""

    def __init__(self, *, max_length: int, actual_length: int) -> None:
        message = (
            f"Input length {actual_length} exceeds maximum allowed {max_length}. "
            "Rejecting to prevent oversized requests."
        )
        super().__init__(message)
        self.max_length = max_length
        self.actual_length = actual_length


class MissingSourceError(InputValidationError):
    """Raised when the input source identifier is absent or blank."""


@dataclass(frozen=True)
class InputPacket:
    """
    Immutable data packet containing validated user input and metadata.

    Attributes:
        text: Normalized user input (trimmed, newlines canonicalized to ``\n``).
        source: Non-empty identifier for where the input originated (e.g., "cli").
        request_id: Unique UUIDv4 identifier for this request.
        timestamp: UTC timestamp when the packet was created.

    Design notes:
    - Frozen to ensure metadata cannot be mutated after creation.
    - Newlines are normalized to keep logs and hashes consistent across platforms.
    - Source is required to aid traceability in safety-critical systems.
    """

    text: str
    source: str
    request_id: uuid.UUID
    timestamp: datetime


class InputInterface:
    """
    Deterministic interface for accepting and validating user input.

    This class enforces safety constraints (non-empty input, length bounds,
    newline normalization, required source) before constructing an immutable
    InputPacket. It performs no I/O and no AI logic.
    """

    def __init__(self, *, max_length: int = 10_000) -> None:
        if max_length <= 0:
            raise ValueError("max_length must be positive")
        self._max_length = max_length

    @property
    def max_length(self) -> int:
        """Maximum allowed input length after normalization."""

        return self._max_length

    def accept(self, raw_text: str, *, source: str) -> InputPacket:
        """
        Validate and normalize raw input into an immutable InputPacket.

        Constraints (enforced deterministically):
        - Reject empty or whitespace-only input to avoid null work items.
        - Enforce a bounded size (default 10_000) to protect resources.
        - Normalize newlines to ``\n`` for cross-platform determinism.
        - Require a non-empty ``source`` for traceability.
        """

        normalized_source = source.strip()
        if not normalized_source:
            raise MissingSourceError("source is required and cannot be empty")

        # Canonicalize newlines before trimming so length checks use normalized form.
        normalized_text = raw_text.replace("\r\n", "\n").replace("\r", "\n")
        trimmed = normalized_text.strip()

        if not trimmed:
            raise EmptyInputError("input must not be empty or whitespace-only")

        if len(trimmed) > self._max_length:
            raise MaxLengthExceededError(max_length=self._max_length, actual_length=len(trimmed))

        request_id = uuid.uuid4()
        timestamp = datetime.now(UTC)

        # Attach metadata safely; InputPacket is frozen to prevent mutation.
        return InputPacket(
            text=trimmed,
            source=normalized_source,
            request_id=request_id,
            timestamp=timestamp,
        )
