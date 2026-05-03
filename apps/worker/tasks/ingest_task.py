"""Document ingest task — Phase 1 pipeline.

Flow:
  [session-1] mark processing
  download file → DoclingProcessor → Embedder   (no DB session held)
  [session-2] VectorStore insert → update Document/Job → mark ready/failed
"""
from __future__ import annotations

import asyncio
import logging
from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime
from uuid import UUID

from sqlalchemy import select

from apps.api.config import get_settings
from apps.api.core import database
from apps.api.models.document import Document
from apps.api.models.job import AsyncJob
from apps.api.services.document_service import DocumentService
from apps.worker.main import celery_app
from aris.ingestion.pymupdf_processor import PyMuPDFProcessor
from aris.ingestion.embedder import Embedder
from aris.retrieval.vector_store import VectorStore

logger = logging.getLogger(__name__)


@celery_app.task(name="apps.worker.tasks.ingest_task.ingest_document")
def ingest_document(job_id: str, document_id: str) -> None:
    """Celery entry point — runs the async pipeline in a dedicated event loop."""
    try:
        asyncio.get_running_loop()
        # Already inside an event loop (eager mode or test context)
        with ThreadPoolExecutor(max_workers=1) as pool:
            pool.submit(asyncio.run, _ingest_async(job_id, document_id)).result()
    except RuntimeError:
        asyncio.run(_ingest_async(job_id, document_id))


async def _ingest_async(job_id: str, document_id: str) -> None:
    settings = get_settings()
    database.init_database(settings)

    if database.SessionLocal is None:
        raise RuntimeError("Database session factory is not initialized")

    doc_service = DocumentService(settings)
    processor = PyMuPDFProcessor()
    embedder = Embedder(
        model=settings.embed_model,
        base_url=settings.embed_base_url or settings.ollama_base_url,
        fallback_to_local=True,
        timeout=90.0,  # generous per-batch timeout; batches are ≤10 chunks each
    )

    # ── Session 1: fetch records & mark in-progress, then close ──────────────
    # Keep session short so it doesn't stay open during multi-minute CPU work.
    async with database.SessionLocal() as session:
        job = await session.scalar(select(AsyncJob).where(AsyncJob.id == UUID(job_id)))
        document = await session.scalar(select(Document).where(Document.id == UUID(document_id)))

        if job is None or document is None:
            logger.warning("ingest_document: job or document not found (job=%s doc=%s)", job_id, document_id)
            await database.dispose_database()
            return

        job.status = "processing"
        job.started_at = datetime.now(UTC)
        document.status = "processing"
        document.error_msg = None
        await session.commit()

        # Extract all values we need as plain Python — session closes after this block.
        doc_id = str(document.id)
        doc_s3_key = document.s3_key
        doc_filename = document.filename
        existing_metadata = dict(document.metadata_json or {})

    # ── CPU-intensive work — no DB session held ───────────────────────────────
    temp_path = None
    error: Exception | None = None
    processed = None
    embeddings: list[list[float]] = []

    try:
        # 1. Download to temp file
        temp_path, used_fallback = doc_service.download_to_tempfile(doc_s3_key, doc_filename)

        # 2. Process with Docling
        if used_fallback:
            processed = processor.process_text(
                temp_path.read_text(encoding="utf-8", errors="replace"),
                source_name=doc_filename,
            )
        else:
            processed = processor.process(temp_path)

        logger.info(
            "Processed document %s → %d chunks (title=%r, authors=%s)",
            doc_id, len(processed.chunks), processed.title, processed.authors,
        )

        # 3. Generate embeddings (potentially slow — Ollama or sentence-transformers)
        chunk_texts = [c.content for c in processed.chunks]
        if chunk_texts:
            try:
                embeddings = embedder.embed_batch(chunk_texts)
            except Exception as emb_exc:
                logger.warning("Embedding failed for document %s: %s", doc_id, emb_exc)
                embeddings = [[] for _ in chunk_texts]

    except Exception as exc:
        logger.exception("Pre-DB processing failed for document %s", doc_id)
        error = exc
    finally:
        if temp_path is not None:
            temp_path.unlink(missing_ok=True)

    # ── Session 2: write results — fresh connection, no stale state ───────────
    async with database.SessionLocal() as session:
        job = await session.scalar(select(AsyncJob).where(AsyncJob.id == UUID(job_id)))
        document = await session.scalar(select(Document).where(Document.id == UUID(document_id)))

        if job is None or document is None:
            logger.error("ingest_document: records disappeared before write (job=%s doc=%s)", job_id, document_id)
            await database.dispose_database()
            return

        try:
            if error is not None or processed is None:
                raise error or RuntimeError("Processing produced no output")

            # 4. Store chunks in pgvector
            chunk_batch = [
                {
                    "document_id": doc_id,
                    "chunk_index": chunk.chunk_index,
                    "content": _clean(chunk.content),
                    "embedding": embeddings[i] if i < len(embeddings) and embeddings[i] else [],
                    "page_number": chunk.page_number,
                    "section": _clean(chunk.section) if chunk.section else chunk.section,
                    "metadata": chunk.metadata,
                }
                for i, chunk in enumerate(processed.chunks)
            ]

            embedded_chunks = [c for c in chunk_batch if c["embedding"]]
            text_only_chunks = [c for c in chunk_batch if not c["embedding"]]

            store = VectorStore(session)
            if embedded_chunks:
                await store.insert_chunk_batch(embedded_chunks)
            for c in text_only_chunks:
                await _insert_chunk_no_embedding(session, c)

            logger.info(
                "Stored %d embedded + %d text-only chunks for document %s",
                len(embedded_chunks), len(text_only_chunks), doc_id,
            )

            # 5. Update Document record
            document.full_text = _clean(processed.full_text)
            document.status = "ready"
            document.metadata_json = {
                **existing_metadata,
                "title": processed.title,
                "authors": processed.authors,
                "page_count": processed.metadata.get("page_count", 0),
                "format": processed.metadata.get("format", ""),
                "chunk_count": len(processed.chunks),
                "embedded_chunk_count": len(embedded_chunks),
                "content_length": len(processed.full_text),
                "content_preview": processed.full_text[:2000],
            }

            job.status = "ready"
            job.error = None
            job.result = {
                "document_id": doc_id,
                "status": "ready",
                "chunk_count": len(processed.chunks),
                "embedded": len(embedded_chunks),
            }

        except Exception as exc:
            logger.exception("Ingest write-phase failed for document %s", doc_id)
            document.status = "failed"
            document.error_msg = str(exc)
            job.status = "failed"
            job.error = str(exc)
            job.result = {"document_id": doc_id, "status": "failed"}

        finally:
            job.completed_at = datetime.now(UTC)
            await session.commit()

    await database.dispose_database()


def _clean(text: str | None) -> str | None:
    """Strip null bytes and other characters PostgreSQL rejects in UTF-8 text columns."""
    if text is None:
        return None
    return text.replace("\x00", "")


async def _insert_chunk_no_embedding(session, chunk: dict) -> None:
    """Insert a chunk row without an embedding vector (NULL embedding column)."""
    from uuid import UUID, uuid4
    from apps.api.models.chunk import DocumentChunk
    from sqlalchemy import select as sa_select

    existing_q = await session.execute(
        sa_select(DocumentChunk).where(
            DocumentChunk.document_id == UUID(str(chunk["document_id"])),
            DocumentChunk.chunk_index == chunk["chunk_index"],
        )
    )
    existing = existing_q.scalar_one_or_none()
    if existing is not None:
        existing.content = chunk["content"]
        existing.section = chunk.get("section")
        existing.metadata_json = chunk.get("metadata") or {}
    else:
        session.add(
            DocumentChunk(
                id=uuid4(),
                document_id=UUID(str(chunk["document_id"])),
                chunk_index=chunk["chunk_index"],
                content=chunk["content"],
                page_number=chunk.get("page_number"),
                section=chunk.get("section"),
                metadata_json=chunk.get("metadata") or {},
            )
        )
