"""
Fast PDF/text processor using PyMuPDF (fitz).

Why this instead of Docling:
  - Docling loads 3 heavy ML models on first use (~1-2 min cold start, 3-5 min/doc)
  - PyMuPDF is a pure-C library: opens, reads, and closes a 20-page PDF in <1 second
  - For text-based PDFs (academic papers) extraction quality is equivalent

Exposes the same ProcessedDocument / ProcessedChunk dataclasses as docling_processor
so the worker task needs no changes beyond swapping the import.
"""
from __future__ import annotations

import logging
import re
from pathlib import Path

from aris.ingestion.docling_processor import (
    ProcessedChunk,
    ProcessedDocument,
    _split_into_chunks,
)

logger = logging.getLogger(__name__)

# Words per chunk (≈512 tokens); must match the DB vector dimension expectations
CHUNK_WORDS = 400
OVERLAP_WORDS = 40

# Section heading patterns common in academic papers
_SECTION_RE = re.compile(
    r"^(?:"
    r"\d+(?:\.\d+)*\.?\s+[A-Z]"   # "1. Introduction", "2.1 Methods"
    r"|[IVX]+\.\s+[A-Z]"           # "I. Introduction" (Roman numerals)
    r"|[A-Z][A-Z\s]{3,40}$"        # "ABSTRACT", "INTRODUCTION" (ALL CAPS)
    r")",
    re.MULTILINE,
)


class PyMuPDFProcessor:
    """Process PDFs and text files into structured chunks using PyMuPDF."""

    def process(self, file_path: str | Path) -> ProcessedDocument:
        path = Path(file_path)
        if path.suffix.lower() == ".txt":
            return self._process_text_file(path)
        try:
            return self._process_pdf(path)
        except Exception as exc:
            logger.warning("PyMuPDF failed for %s (%s) — falling back to plain-text", path.name, exc)
            return self._process_text_file(path)

    def process_text(self, text: str, source_name: str = "text") -> ProcessedDocument:
        """Process a plain-text string directly (used by worker fallback path)."""
        raw = _split_into_chunks(text, CHUNK_WORDS, OVERLAP_WORDS)
        chunks = [ProcessedChunk(chunk_index=i, content=c) for i, c in enumerate(raw) if c.strip()]
        return ProcessedDocument(source_path=source_name, chunks=chunks)

    # ── private ──────────────────────────────────────────────────────────────

    def _process_pdf(self, path: Path) -> ProcessedDocument:
        import fitz  # pymupdf — imported lazily so non-PDF paths skip it

        doc = fitz.open(str(path))
        try:
            meta = doc.metadata or {}
            page_count = len(doc)

            # PDF metadata title / authors
            title: str | None = (meta.get("title") or "").strip() or None
            author_raw = (meta.get("author") or "").strip()
            authors = [a.strip() for a in re.split(r"[,;]", author_raw) if a.strip()]

            # Extract per-page text
            pages: list[tuple[int, str]] = []
            for i in range(page_count):
                text = doc[i].get_text("text").strip()
                if text:
                    pages.append((i + 1, text))
        finally:
            doc.close()

        if not pages:
            raise ValueError("No text extracted — PDF may be image-only")

        # Heuristic title from first page when metadata is blank
        if not title:
            title = _extract_title_heuristic(pages[0][1], path.stem)

        chunks = _chunk_pages(pages, CHUNK_WORDS, OVERLAP_WORDS)

        return ProcessedDocument(
            source_path=str(path),
            chunks=chunks,
            title=title,
            authors=authors,
            metadata={"page_count": page_count, "format": "pdf"},
        )

    def _process_text_file(self, path: Path) -> ProcessedDocument:
        try:
            text = path.read_text(encoding="utf-8", errors="replace")
        except Exception as exc:
            text = f"[Failed to read: {exc}]"
        raw = _split_into_chunks(text, CHUNK_WORDS, OVERLAP_WORDS)
        chunks = [ProcessedChunk(chunk_index=i, content=c.strip()) for i, c in enumerate(raw) if c.strip()]
        return ProcessedDocument(
            source_path=str(path),
            chunks=chunks,
            title=path.stem,
            metadata={"format": path.suffix.lstrip(".")},
        )


# ── helpers ───────────────────────────────────────────────────────────────────

def _extract_title_heuristic(first_page_text: str, fallback: str) -> str:
    """Best-effort title from the first page: first non-trivial line."""
    for line in first_page_text.splitlines():
        line = line.strip()
        # Skip very short lines, page numbers, and URLs
        if len(line) < 12 or line.isdigit() or line.startswith("http"):
            continue
        # Stop at lines that look like author lists or affiliations
        if re.search(r"\d{4}", line) and len(line) < 20:
            continue
        return line[:300]
    return fallback


def _chunk_pages(
    pages: list[tuple[int, str]],
    max_words: int,
    overlap_words: int,
) -> list[ProcessedChunk]:
    """
    Chunk text extracted from multiple pages.

    Tracks which page each chunk originated from so the DB row has accurate
    page_number (unlike Docling's proportional approximation).
    """
    # Detect section heading to populate chunk.section
    chunks: list[ProcessedChunk] = []
    chunk_index = 0
    current_words: list[str] = []
    current_page: int = 1
    current_section: str | None = None
    word_pages: list[int] = []  # parallel list: which page each word came from

    for page_num, page_text in pages:
        paragraphs = [p.strip() for p in re.split(r"\n\n+|\n(?=\s*\n)", page_text) if p.strip()]
        for para in paragraphs:
            # Detect section heading
            if _SECTION_RE.match(para) and len(para.split()) <= 12:
                current_section = para.strip()

            para_words = para.split()
            para_pages = [page_num] * len(para_words)

            # If adding this paragraph would overflow, flush first
            if len(current_words) + len(para_words) > max_words and current_words:
                chunks.append(ProcessedChunk(
                    chunk_index=chunk_index,
                    content=" ".join(current_words),
                    page_number=word_pages[0] if word_pages else current_page,
                    section=current_section,
                ))
                chunk_index += 1
                # Keep overlap
                current_words = current_words[-overlap_words:] if overlap_words else []
                word_pages = word_pages[-overlap_words:] if overlap_words else []

            # If single paragraph exceeds max_words, split it directly
            if len(para_words) > max_words:
                if current_words:
                    chunks.append(ProcessedChunk(
                        chunk_index=chunk_index,
                        content=" ".join(current_words),
                        page_number=word_pages[0] if word_pages else page_num,
                        section=current_section,
                    ))
                    chunk_index += 1
                    current_words = []
                    word_pages = []
                start = 0
                while start < len(para_words):
                    end = start + max_words
                    chunks.append(ProcessedChunk(
                        chunk_index=chunk_index,
                        content=" ".join(para_words[start:end]),
                        page_number=para_pages[start],
                        section=current_section,
                    ))
                    chunk_index += 1
                    start += max_words - overlap_words
                current_words = para_words[max(0, len(para_words) - overlap_words):]
                word_pages = para_pages[max(0, len(para_pages) - overlap_words):]
                continue

            current_words.extend(para_words)
            word_pages.extend(para_pages)

    # Flush remainder (minimum 30 words to avoid tiny trailing chunks)
    if len(current_words) >= 30:
        chunks.append(ProcessedChunk(
            chunk_index=chunk_index,
            content=" ".join(current_words),
            page_number=word_pages[0] if word_pages else 1,
            section=current_section,
        ))

    return chunks
