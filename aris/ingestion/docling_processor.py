"""
Docling-based PDF and text processor.

Converts documents into structured ProcessedChunk lists, preserving section
headings and page numbers. Falls back to plain-text chunking when Docling
cannot parse a file (corrupt PDFs, unsupported formats).

First import is slow (~3-5s) because Docling initialises its ML models.
Subsequent calls are fast.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class ProcessedChunk:
    chunk_index: int
    content: str
    page_number: int | None = None
    section: str | None = None
    metadata: dict = field(default_factory=dict)


@dataclass
class ProcessedDocument:
    source_path: str
    chunks: list[ProcessedChunk]
    title: str | None = None
    authors: list[str] = field(default_factory=list)
    metadata: dict = field(default_factory=dict)

    @property
    def full_text(self) -> str:
        return "\n\n".join(c.content for c in self.chunks)


class DoclingProcessor:
    """Process PDFs and text files into structured chunks."""

    # Words per chunk target (≈ 512 tokens)
    CHUNK_WORDS = 400
    # Overlap between consecutive chunks
    OVERLAP_WORDS = 40

    def process(self, file_path: str | Path) -> ProcessedDocument:
        """Convert a file to ProcessedDocument. Tries Docling first, falls back to plain text."""
        path = Path(file_path)
        suffix = path.suffix.lower()

        if suffix == ".txt":
            return self._process_text_file(path)

        try:
            return self._process_with_docling(path)
        except Exception:
            # Docling failed (corrupt PDF, unsupported format, etc.) — degrade gracefully
            return self._process_text_file(path)

    def process_text(self, text: str, source_name: str = "text") -> ProcessedDocument:
        """Process plain text content directly."""
        raw_chunks = _split_into_chunks(text, self.CHUNK_WORDS, self.OVERLAP_WORDS)
        chunks = [
            ProcessedChunk(chunk_index=i, content=c, section=None)
            for i, c in enumerate(raw_chunks)
        ]
        return ProcessedDocument(source_path=source_name, chunks=chunks)

    # ── private ──────────────────────────────────────────────────────────────

    def _process_with_docling(self, path: Path) -> ProcessedDocument:
        from docling.document_converter import DocumentConverter

        converter = DocumentConverter()
        result = converter.convert(str(path))
        doc = result.document

        title = self._extract_title(doc, path)
        authors = self._extract_authors(doc)
        page_count = len(doc.pages) if hasattr(doc, "pages") and doc.pages else 0

        # Export to markdown — preserves heading structure as ## / ### markers
        md_text = doc.export_to_markdown()

        # Build chunks from the structured markdown
        section_chunks = _split_markdown_by_section(md_text)
        chunks: list[ProcessedChunk] = []
        idx = 0
        for section_title, section_text in section_chunks:
            sub_chunks = _split_into_chunks(section_text, self.CHUNK_WORDS, self.OVERLAP_WORDS)
            for text in sub_chunks:
                if not text.strip():
                    continue
                chunks.append(ProcessedChunk(
                    chunk_index=idx,
                    content=text.strip(),
                    section=section_title,
                ))
                idx += 1

        # Assign approximate page numbers by position in document
        _assign_page_numbers(chunks, page_count)

        return ProcessedDocument(
            source_path=str(path),
            chunks=chunks,
            title=title,
            authors=authors,
            metadata={"page_count": page_count, "format": path.suffix.lstrip(".")},
        )

    def _process_text_file(self, path: Path) -> ProcessedDocument:
        try:
            text = path.read_text(encoding="utf-8", errors="replace")
        except Exception as exc:
            text = f"[Failed to read file: {exc}]"

        raw_chunks = _split_into_chunks(text, self.CHUNK_WORDS, self.OVERLAP_WORDS)
        chunks = [
            ProcessedChunk(chunk_index=i, content=c.strip())
            for i, c in enumerate(raw_chunks)
            if c.strip()
        ]
        return ProcessedDocument(
            source_path=str(path),
            chunks=chunks,
            title=path.stem,
            metadata={"format": path.suffix.lstrip(".")},
        )

    @staticmethod
    def _extract_title(doc, path: Path) -> str:
        try:
            # Docling may surface a title from PDF metadata
            if hasattr(doc, "name") and doc.name:
                candidate = doc.name.strip()
                if len(candidate) > 4 and not candidate.lower().endswith(".pdf"):
                    return candidate
        except Exception:
            pass
        return path.stem

    @staticmethod
    def _extract_authors(doc) -> list[str]:
        """Best-effort author extraction from docling document metadata."""
        try:
            if hasattr(doc, "meta") and doc.meta:
                meta = doc.meta
                if hasattr(meta, "authors") and meta.authors:
                    return [str(a) for a in meta.authors]
        except Exception:
            pass
        return []


# ── helpers ──────────────────────────────────────────────────────────────────

def _split_markdown_by_section(md_text: str) -> list[tuple[str | None, str]]:
    """
    Split markdown into (section_heading, text) pairs.
    Splits at top-level (##) and second-level (###) headings.
    Returns [(heading, body_text), ...].
    """
    # Match ## Heading or ### Heading lines
    heading_re = re.compile(r"^(#{1,3})\s+(.+)$", re.MULTILINE)
    sections: list[tuple[str | None, str]] = []
    last_end = 0
    current_heading: str | None = None

    for match in heading_re.finditer(md_text):
        body = md_text[last_end : match.start()].strip()
        if body:
            sections.append((current_heading, body))
        current_heading = match.group(2).strip()
        last_end = match.end()

    # trailing text after last heading
    tail = md_text[last_end:].strip()
    if tail:
        sections.append((current_heading, tail))

    # If no headings found, treat entire text as one section
    if not sections:
        sections = [(None, md_text.strip())]

    return sections


def _split_into_chunks(text: str, max_words: int, overlap_words: int) -> list[str]:
    """
    Sliding-window word-based chunking.
    Splits on paragraph boundaries (\\n\\n) first, then by word count.
    """
    paragraphs = [p.strip() for p in re.split(r"\n\n+", text) if p.strip()]
    if not paragraphs:
        return []

    chunks: list[str] = []
    current_words: list[str] = []

    def flush() -> None:
        if current_words:
            chunks.append(" ".join(current_words))

    for para in paragraphs:
        para_words = para.split()

        # If this paragraph alone exceeds max_words, split it by words directly
        if len(para_words) > max_words:
            # flush what we have first
            flush()
            current_words = []
            start = 0
            while start < len(para_words):
                chunk_words = para_words[start : start + max_words]
                chunks.append(" ".join(chunk_words))
                start += max_words - overlap_words
            # carry overlap into next paragraph
            if chunks:
                current_words = chunks[-1].split()[-overlap_words:]
            continue

        if len(current_words) + len(para_words) > max_words:
            flush()
            # start next chunk with overlap from end of current
            current_words = current_words[-overlap_words:] if overlap_words else []

        current_words.extend(para_words)

    flush()
    return chunks


def _assign_page_numbers(chunks: list[ProcessedChunk], page_count: int) -> None:
    """Distribute page numbers proportionally across chunks."""
    if page_count <= 0 or not chunks:
        return
    total = len(chunks)
    for i, chunk in enumerate(chunks):
        chunk.page_number = max(1, round((i / total) * page_count) + 1)
