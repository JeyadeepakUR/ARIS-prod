"""
Public API for text chunking.

DoclingProcessor uses this internally for plain-text paths.
Other parts of the codebase (e.g. worker tasks) can call chunk_document()
directly when they already have extracted text.
"""
from __future__ import annotations

from aris.ingestion.docling_processor import ProcessedChunk, _split_into_chunks

_DEFAULT_MAX_WORDS = 400
_DEFAULT_OVERLAP = 40


def chunk_document(
    text: str,
    *,
    max_words: int = _DEFAULT_MAX_WORDS,
    overlap_words: int = _DEFAULT_OVERLAP,
    section: str | None = None,
) -> list[ProcessedChunk]:
    """
    Split text into overlapping chunks.

    Uses the same sliding-window algorithm as DoclingProcessor.
    Returns ProcessedChunk list with sequential chunk_index values.
    """
    raw = _split_into_chunks(text, max_words, overlap_words)
    return [
        ProcessedChunk(chunk_index=i, content=c.strip(), section=section)
        for i, c in enumerate(raw)
        if c.strip()
    ]
