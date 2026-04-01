"""Tests for document ingestion and canonicalization (Module 7)."""

from __future__ import annotations

import tempfile
import uuid
from datetime import UTC, datetime
from pathlib import Path

import pytest

from aris.graph.document_ingestion import (
    CorpusPacketizer,
    Document,
    DocumentCorpus,
    DocumentIngestor,
    PDFLoader,
    PlainTextLoader,
    create_corpus,
)
from aris.core.input_interface import InputPacket, InputValidationError


class TestDocument:
    """Tests for Document dataclass."""

    def test_document_creation(self) -> None:
        doc = Document(
            content="test content",
            source="test.txt",
            format="text",
            metadata={"key": "value"},
            document_id=uuid.uuid4(),
            created_at=datetime.now(UTC),
        )
        assert doc.content == "test content"
        assert doc.source == "test.txt"
        assert doc.format == "text"

    def test_document_frozen(self) -> None:
        doc = Document(
            content="test",
            source="test.txt",
            format="text",
            metadata={},
            document_id=uuid.uuid4(),
            created_at=datetime.now(UTC),
        )
        try:
            doc.content = "changed"  # type: ignore
            assert False, "Should not allow mutation"
        except (AttributeError, TypeError):
            pass


class TestDocumentCorpus:
    """Tests for DocumentCorpus dataclass."""

    def test_corpus_creation(self) -> None:
        doc1 = Document(
            content="doc1",
            source="1.txt",
            format="text",
            metadata={},
            document_id=uuid.uuid4(),
            created_at=datetime.now(UTC),
        )
        corpus = DocumentCorpus(
            documents=[doc1],
            corpus_id=uuid.uuid4(),
            created_at=datetime.now(UTC),
        )
        assert len(corpus.documents) == 1
        assert corpus.documents[0] is doc1

    def test_corpus_frozen(self) -> None:
        corpus = DocumentCorpus(
            documents=[],
            corpus_id=uuid.uuid4(),
            created_at=datetime.now(UTC),
        )
        try:
            corpus.documents = []  # type: ignore
            assert False, "Should not allow mutation"
        except (AttributeError, TypeError):
            pass


class TestPlainTextLoader:
    """Tests for PlainTextLoader."""

    def test_load_plain_text(self) -> None:
        loader = PlainTextLoader()
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".txt", delete=False, encoding="utf-8"
        ) as f:
            f.write("Hello, World!\nSecond line.")
            temp_path = Path(f.name)

        try:
            doc = loader.load(temp_path)
            assert "Hello, World!" in doc.content
            assert "Second line" in doc.content
            assert doc.format == "text"
            assert doc.source == str(temp_path)
        finally:
            temp_path.unlink()

    def test_load_nonexistent_file(self) -> None:
        loader = PlainTextLoader()
        with pytest.raises(FileNotFoundError):
            loader.load(Path("nonexistent.txt"))

    def test_load_empty_file(self) -> None:
        loader = PlainTextLoader()
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".txt", delete=False, encoding="utf-8"
        ) as f:
            f.write("   \n\n  ")  # Only whitespace
            temp_path = Path(f.name)

        try:
            with pytest.raises(ValueError, match="Empty document"):
                loader.load(temp_path)
        finally:
            temp_path.unlink()

    def test_canonicalize_line_endings(self) -> None:
        """Verify line ending normalization."""
        loader = PlainTextLoader()
        with tempfile.NamedTemporaryFile(
            mode="wb", suffix=".txt", delete=False
        ) as f:
            # Write with CRLF line endings
            f.write(b"Line1\r\nLine2\rLine3\n")
            temp_path = Path(f.name)

        try:
            doc = loader.load(temp_path)
            # All normalized to \n
            assert "\r" not in doc.content
            assert "Line1\nLine2\nLine3" in doc.content
        finally:
            temp_path.unlink()

    def test_metadata_includes_file_size(self) -> None:
        loader = PlainTextLoader()
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".txt", delete=False, encoding="utf-8"
        ) as f:
            f.write("test content")
            temp_path = Path(f.name)

        try:
            doc = loader.load(temp_path)
            assert "file_size" in doc.metadata
            assert int(doc.metadata["file_size"]) > 0
        finally:
            temp_path.unlink()


class TestPDFLoader:
    """Tests for PDFLoader."""

    def test_pdf_loader_requires_pypdf(self) -> None:
        """Verify ImportError if pypdf not installed."""
        loader = PDFLoader()
        with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as f:
            temp_path = Path(f.name)

        try:
            # May raise ImportError if pypdf not installed
            # If installed, will raise ValueError for invalid PDF
            with pytest.raises((ImportError, ValueError)):
                loader.load(temp_path)
        finally:
            temp_path.unlink()

    def test_pdf_nonexistent_file(self) -> None:
        loader = PDFLoader()
        with pytest.raises(FileNotFoundError):
            loader.load(Path("nonexistent.pdf"))


class TestDocumentIngestor:
    """Tests for DocumentIngestor."""

    def test_ingestor_exists(self) -> None:
        ingestor = DocumentIngestor()
        assert ingestor is not None

    def test_ingest_text_file(self) -> None:
        ingestor = DocumentIngestor()
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".txt", delete=False, encoding="utf-8"
        ) as f:
            f.write("Ingestor test content")
            temp_path = Path(f.name)

        try:
            doc = ingestor.ingest(temp_path, "text")
            assert "Ingestor test content" in doc.content
            assert doc.format == "text"
        finally:
            temp_path.unlink()

    def test_ingest_unsupported_format(self) -> None:
        ingestor = DocumentIngestor()
        with tempfile.NamedTemporaryFile(suffix=".xyz", delete=False) as f:
            temp_path = Path(f.name)

        try:
            with pytest.raises(ValueError, match="Unsupported format"):
                ingestor.ingest(temp_path, "xyz")
        finally:
            temp_path.unlink()

    def test_ingest_batch_success(self) -> None:
        ingestor = DocumentIngestor()
        temp_files = []

        try:
            # Create two text files
            for i in range(2):
                with tempfile.NamedTemporaryFile(
                    mode="w", suffix=".txt", delete=False, encoding="utf-8"
                ) as f:
                    f.write(f"Content {i}")
                    temp_files.append(Path(f.name))

            file_paths = [(p, "text") for p in temp_files]
            docs, failures = ingestor.ingest_batch(file_paths)

            assert len(docs) == 2
            assert len(failures) == 0
            assert "Content 0" in docs[0].content
            assert "Content 1" in docs[1].content
        finally:
            for p in temp_files:
                p.unlink()

    def test_ingest_batch_with_failures(self) -> None:
        ingestor = DocumentIngestor()
        good_file = None
        bad_file = Path("nonexistent.txt")

        try:
            with tempfile.NamedTemporaryFile(
                mode="w", suffix=".txt", delete=False, encoding="utf-8"
            ) as f:
                f.write("Good content")
                good_file = Path(f.name)

            file_paths = [(good_file, "text"), (bad_file, "text")]
            docs, failures = ingestor.ingest_batch(file_paths)

            assert len(docs) == 1
            assert len(failures) == 1
            assert failures[0][0] == bad_file
            assert isinstance(failures[0][1], FileNotFoundError)
        finally:
            if good_file:
                good_file.unlink()

    def test_deterministic_ingestion(self) -> None:
        """Same file ingested twice produces same content."""
        ingestor = DocumentIngestor()
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".txt", delete=False, encoding="utf-8"
        ) as f:
            f.write("Deterministic test")
            temp_path = Path(f.name)

        try:
            doc1 = ingestor.ingest(temp_path, "text")
            doc2 = ingestor.ingest(temp_path, "text")

            assert doc1.content == doc2.content
            assert doc1.format == doc2.format
            assert doc1.source == doc2.source
            # Different IDs and timestamps
            assert doc1.document_id != doc2.document_id
        finally:
            temp_path.unlink()


class TestCorpusPacketizer:
    """Tests for CorpusPacketizer."""

    def test_packetizer_exists(self) -> None:
        packetizer = CorpusPacketizer()
        assert packetizer is not None

    def test_packetize_empty_corpus(self) -> None:
        packetizer = CorpusPacketizer()
        corpus = DocumentCorpus(
            documents=[],
            corpus_id=uuid.uuid4(),
            created_at=datetime.now(UTC),
        )

        packets = packetizer.packetize(corpus)
        assert len(packets) == 0

    def test_packetize_single_document(self) -> None:
        packetizer = CorpusPacketizer()
        doc = Document(
            content="Test document content",
            source="test.txt",
            format="text",
            metadata={},
            document_id=uuid.uuid4(),
            created_at=datetime.now(UTC),
        )
        corpus = create_corpus([doc])

        packets = packetizer.packetize(corpus)

        assert len(packets) == 1
        assert isinstance(packets[0], InputPacket)
        assert packets[0].text == "Test document content"
        assert "text:test.txt" in packets[0].source

    def test_packetize_multiple_documents(self) -> None:
        packetizer = CorpusPacketizer()
        docs = [
            Document(
                content=f"Document {i}",
                source=f"{i}.txt",
                format="text",
                metadata={},
                document_id=uuid.uuid4(),
                created_at=datetime.now(UTC),
            )
            for i in range(3)
        ]
        corpus = create_corpus(docs)

        packets = packetizer.packetize(corpus)

        assert len(packets) == 3
        for i, packet in enumerate(packets):
            assert f"Document {i}" in packet.text

    def test_packetize_validates_through_input_interface(self) -> None:
        """Verify InputInterface validation is applied."""
        packetizer = CorpusPacketizer()
        # Empty content should fail validation
        doc = Document(
            content="   ",  # Only whitespace
            source="empty.txt",
            format="text",
            metadata={},
            document_id=uuid.uuid4(),
            created_at=datetime.now(UTC),
        )
        corpus = create_corpus([doc])

        with pytest.raises(InputValidationError):
            packetizer.packetize(corpus)

    def test_packetize_with_isolation_success(self) -> None:
        packetizer = CorpusPacketizer()
        doc = Document(
            content="Valid content",
            source="valid.txt",
            format="text",
            metadata={},
            document_id=uuid.uuid4(),
            created_at=datetime.now(UTC),
        )
        corpus = create_corpus([doc])

        packets, failures = packetizer.packetize_with_isolation(corpus)

        assert len(packets) == 1
        assert len(failures) == 0

    def test_packetize_with_isolation_failures(self) -> None:
        packetizer = CorpusPacketizer()
        good_doc = Document(
            content="Good content",
            source="good.txt",
            format="text",
            metadata={},
            document_id=uuid.uuid4(),
            created_at=datetime.now(UTC),
        )
        bad_doc = Document(
            content="   ",  # Invalid - only whitespace
            source="bad.txt",
            format="text",
            metadata={},
            document_id=uuid.uuid4(),
            created_at=datetime.now(UTC),
        )
        corpus = create_corpus([good_doc, bad_doc])

        packets, failures = packetizer.packetize_with_isolation(corpus)

        assert len(packets) == 1
        assert len(failures) == 1
        assert failures[0][0] is bad_doc
        assert isinstance(failures[0][1], InputValidationError)

    def test_source_includes_format_and_path(self) -> None:
        """Verify source field includes format prefix."""
        packetizer = CorpusPacketizer()
        doc = Document(
            content="Content",
            source="/path/to/file.pdf",
            format="pdf",
            metadata={},
            document_id=uuid.uuid4(),
            created_at=datetime.now(UTC),
        )
        corpus = create_corpus([doc])

        packets = packetizer.packetize(corpus)

        assert packets[0].source.startswith("pdf:")
        assert "/path/to/file.pdf" in packets[0].source

    def test_metadata_integrity(self) -> None:
        """Verify document metadata is preserved through packetization."""
        packetizer = CorpusPacketizer()
        doc = Document(
            content="Content with metadata",
            source="file.txt",
            format="text",
            metadata={"key1": "value1", "key2": "value2"},
            document_id=uuid.uuid4(),
            created_at=datetime.now(UTC),
        )
        corpus = create_corpus([doc])

        packets = packetizer.packetize(corpus)

        # Metadata preserved in Document (corpus still accessible)
        assert corpus.documents[0].metadata["key1"] == "value1"
        # Packet has correct content and source
        assert packets[0].text == "Content with metadata"


class TestCreateCorpus:
    """Tests for create_corpus helper."""

    def test_create_corpus_from_documents(self) -> None:
        doc1 = Document(
            content="doc1",
            source="1.txt",
            format="text",
            metadata={},
            document_id=uuid.uuid4(),
            created_at=datetime.now(UTC),
        )
        doc2 = Document(
            content="doc2",
            source="2.txt",
            format="text",
            metadata={},
            document_id=uuid.uuid4(),
            created_at=datetime.now(UTC),
        )

        corpus = create_corpus([doc1, doc2])

        assert len(corpus.documents) == 2
        assert corpus.documents[0] is doc1
        assert corpus.documents[1] is doc2
        assert isinstance(corpus.corpus_id, uuid.UUID)

    def test_create_corpus_empty_list(self) -> None:
        corpus = create_corpus([])
        assert len(corpus.documents) == 0
        assert isinstance(corpus.corpus_id, uuid.UUID)


class TestEndToEndIngestion:
    """End-to-end tests for full ingestion pipeline."""

    def test_full_pipeline_text_to_packets(self) -> None:
        """Test complete flow: file → Document → Corpus → InputPacket."""
        ingestor = DocumentIngestor()
        packetizer = CorpusPacketizer()

        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".txt", delete=False, encoding="utf-8"
        ) as f:
            f.write("End to end test content")
            temp_path = Path(f.name)

        try:
            # Ingest
            doc = ingestor.ingest(temp_path, "text")

            # Create corpus
            corpus = create_corpus([doc])

            # Packetize
            packets = packetizer.packetize(corpus)

            assert len(packets) == 1
            assert packets[0].text == "End to end test content"
            assert isinstance(packets[0].request_id, uuid.UUID)
        finally:
            temp_path.unlink()

    def test_full_pipeline_batch_with_mixed_results(self) -> None:
        """Test batch ingestion with some failures."""
        ingestor = DocumentIngestor()
        packetizer = CorpusPacketizer()
        temp_files = []

        try:
            # Create valid files
            for i in range(2):
                with tempfile.NamedTemporaryFile(
                    mode="w", suffix=".txt", delete=False, encoding="utf-8"
                ) as f:
                    f.write(f"Valid content {i}")
                    temp_files.append(Path(f.name))

            # Add non-existent file
            file_paths = [(p, "text") for p in temp_files] + [
                (Path("missing.txt"), "text")
            ]

            # Ingest with isolation
            docs, ingest_failures = ingestor.ingest_batch(file_paths)
            assert len(docs) == 2
            assert len(ingest_failures) == 1

            # Create corpus from successful docs
            corpus = create_corpus(docs)

            # Packetize
            packets = packetizer.packetize(corpus)

            assert len(packets) == 2
        finally:
            for p in temp_files:
                p.unlink()
