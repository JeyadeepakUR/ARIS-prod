"""
Document ingestion and canonicalization for ARIS.

Provides format-specific loaders (plain text, PDF) and a canonicalization
pipeline that produces validated documents with provenance metadata.
"""

import uuid
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
import re

from aris.core.input_interface import InputInterface, InputPacket


@dataclass(frozen=True)
class Document:
    """
    Immutable document with content and provenance metadata.

    Attributes:
        content: Extracted text content (canonicalized)
        source: File path or identifier
        format: Document format ("text", "pdf", etc.)
        metadata: Additional metadata (file size, page count, etc.)
        document_id: Unique identifier for this document
        created_at: UTC timestamp when document was ingested
    """

    content: str
    source: str
    format: str
    metadata: dict[str, str]
    document_id: uuid.UUID
    created_at: datetime


@dataclass(frozen=True)
class DocumentCorpus:
    """
    Immutable collection of documents.

    Attributes:
        documents: List of Document objects
        corpus_id: Unique identifier for this corpus
        created_at: UTC timestamp when corpus was created
    """

    documents: list[Document]
    corpus_id: uuid.UUID
    created_at: datetime


class PlainTextLoader:
    """Loads plain text files with UTF-8 encoding."""

    def load(self, file_path: Path) -> Document:
        """
        Load a plain text file.

        Args:
            file_path: Path to the text file

        Returns:
            Document with extracted content

        Raises:
            FileNotFoundError: If file doesn't exist
            ValueError: If file is empty or unreadable
        """
        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")

        try:
            content = file_path.read_text(encoding="utf-8")
        except Exception as e:
            raise ValueError(f"Failed to read file: {e}") from e

        if not content.strip():
            raise ValueError(f"Empty document: {file_path}")

        # Canonicalize: normalize line endings and strip
        content = content.replace("\r\n", "\n").replace("\r", "\n").strip()

        metadata = {
            "file_size": str(file_path.stat().st_size),
            "encoding": "utf-8",
        }

        return Document(
            content=content,
            source=str(file_path),
            format="text",
            metadata=metadata,
            document_id=uuid.uuid4(),
            created_at=datetime.now(UTC),
        )


class PDFLoader:
    """Loads PDF files and extracts text content."""

    _MIN_MEANINGFUL_CHARS = 8

    def load(self, file_path: Path) -> Document:
        """
        Load a PDF file and extract text.

        Args:
            file_path: Path to the PDF file

        Returns:
            Document with extracted text content

        Raises:
            FileNotFoundError: If file doesn't exist
            ValueError: If PDF is empty, corrupted, or unreadable
            ImportError: If pypdf is not installed
        """
        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")

        try:
            from pypdf import PdfReader  # type: ignore
        except ImportError as e:
            raise ImportError(
                "pypdf is required for PDF loading. Install with: pip install pypdf"
            ) from e

        try:
            reader = PdfReader(file_path)
            if getattr(reader, "is_encrypted", False):
                try:
                    reader.decrypt("")
                except Exception as e:
                    raise ValueError(f"Failed to decrypt PDF: {e}") from e

            pages: list[str] = []
            failed_pages: list[str] = []

            for idx, page in enumerate(reader.pages, start=1):
                try:
                    text = self._extract_page_text(page)
                except Exception:
                    text = ""

                if self._is_meaningful_text(text):
                    pages.append(self._canonicalize_pdf_text(text))
                else:
                    failed_pages.append(str(idx))

            if not pages:
                raise ValueError(f"No extractable text in PDF: {file_path}")

            content = "\n\n".join(pages)
            content = self._canonicalize_pdf_text(content)

            if not content:
                raise ValueError(f"Empty document after extraction: {file_path}")

            metadata = {
                "file_size": str(file_path.stat().st_size),
                "page_count": str(len(reader.pages)),
                "extracted_pages": str(len(pages)),
                "failed_pages": ",".join(failed_pages),
                "extraction_ratio": f"{len(pages)}/{len(reader.pages)}",
                "extracted_characters": str(len(content)),
            }

            return Document(
                content=content,
                source=str(file_path),
                format="pdf",
                metadata=metadata,
                document_id=uuid.uuid4(),
                created_at=datetime.now(UTC),
            )

        except Exception as e:
            if isinstance(e, ValueError):
                raise
            raise ValueError(f"Failed to load PDF: {e}") from e

    def _extract_page_text(self, page: object) -> str:
        """Extract text with multiple strategies to handle diverse PDF layouts."""

        extract_text = getattr(page, "extract_text", None)
        if extract_text is None:
            return ""

        primary = extract_text()
        if self._is_meaningful_text(primary):
            return primary

        try:
            layout = extract_text(extraction_mode="layout")
            if self._is_meaningful_text(layout):
                return layout
        except TypeError:
            # Older pypdf versions do not accept extraction_mode.
            pass

        return primary or ""

    @staticmethod
    def _is_meaningful_text(text: str | None) -> bool:
        if not text:
            return False
        alnum_count = sum(1 for char in text if char.isalnum())
        if alnum_count >= PDFLoader._MIN_MEANINGFUL_CHARS:
            return True
        word_count = len([token for token in text.split() if token.strip()])
        return word_count >= 2

    @staticmethod
    def _canonicalize_pdf_text(text: str) -> str:
        """Normalize extracted text to reduce artifacts from PDF layout."""

        normalized = text.replace("\x00", "")
        normalized = normalized.replace("\r\n", "\n").replace("\r", "\n")
        normalized = re.sub(r"(?<=\w)-\n(?=\w)", "", normalized)
        normalized = re.sub(r"[\t\f\v]+", " ", normalized)
        normalized = re.sub(r"[ ]{2,}", " ", normalized)
        normalized = re.sub(r"\n{3,}", "\n\n", normalized)
        normalized = "\n".join(line.strip() for line in normalized.split("\n"))
        return normalized.strip()


class DocumentIngestor:
    """
    Ingests documents using format-specific loaders.

    No auto-detection: format must be explicitly specified.
    """

    def __init__(self) -> None:
        """Initialize ingestor with format-specific loaders."""
        self._loaders: dict[str, PlainTextLoader | PDFLoader] = {
            "text": PlainTextLoader(),
            "pdf": PDFLoader(),
        }

    def ingest(self, file_path: Path, format: str) -> Document:
        """
        Ingest a document with explicit format.

        Args:
            file_path: Path to the document file
            format: Document format ("text" or "pdf")

        Returns:
            Document with extracted content and metadata

        Raises:
            ValueError: If format is unsupported or document is invalid
            FileNotFoundError: If file doesn't exist
        """
        if format not in self._loaders:
            supported = ", ".join(self._loaders.keys())
            raise ValueError(
                f"Unsupported format: {format}. Supported: {supported}"
            )

        loader = self._loaders[format]
        return loader.load(file_path)

    def ingest_batch(
        self, file_paths: list[tuple[Path, str]]
    ) -> tuple[list[Document], list[tuple[Path, Exception]]]:
        """
        Ingest multiple documents with failure isolation.

        Args:
            file_paths: List of (file_path, format) tuples

        Returns:
            Tuple of (successful_documents, failed_documents_with_exceptions)
        """
        documents: list[Document] = []
        failures: list[tuple[Path, Exception]] = []

        for file_path, format in file_paths:
            try:
                doc = self.ingest(file_path, format)
                documents.append(doc)
            except Exception as e:
                failures.append((file_path, e))

        return documents, failures


class CorpusPacketizer:
    """
    Converts DocumentCorpus to InputPacket objects.

    Reuses InputInterface validation to ensure packets meet ARIS requirements.
    """

    def __init__(self) -> None:
        """Initialize packetizer with InputInterface validator."""
        self._input_interface = InputInterface()

    def packetize(self, corpus: DocumentCorpus) -> list[InputPacket]:
        """
        Convert each document in corpus to an InputPacket.

        Each document becomes one packet with document metadata preserved
        in the source field.

        Args:
            corpus: DocumentCorpus to packetize

        Returns:
            List of validated InputPacket objects

        Raises:
            ValidationError: If any document content fails InputInterface validation
        """
        packets: list[InputPacket] = []

        for doc in corpus.documents:
            # Source includes document metadata
            source = f"{doc.format}:{doc.source}"

            # Validate through InputInterface
            packet = self._input_interface.accept(doc.content, source=source)
            packets.append(packet)

        return packets

    def packetize_with_isolation(
        self, corpus: DocumentCorpus
    ) -> tuple[list[InputPacket], list[tuple[Document, Exception]]]:
        """
        Packetize with failure isolation.

        Args:
            corpus: DocumentCorpus to packetize

        Returns:
            Tuple of (successful_packets, failed_documents_with_exceptions)
        """
        packets: list[InputPacket] = []
        failures: list[tuple[Document, Exception]] = []

        for doc in corpus.documents:
            try:
                source = f"{doc.format}:{doc.source}"
                packet = self._input_interface.accept(doc.content, source=source)
                packets.append(packet)
            except Exception as e:
                failures.append((doc, e))

        return packets, failures


def create_corpus(documents: list[Document]) -> DocumentCorpus:
    """
    Create a DocumentCorpus from a list of documents.

    Args:
        documents: List of Document objects

    Returns:
        DocumentCorpus with unique ID and timestamp
    """
    return DocumentCorpus(
        documents=documents,
        corpus_id=uuid.uuid4(),
        created_at=datetime.now(UTC),
    )
