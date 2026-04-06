"""Document ingest task for Sprint 2 async pipeline."""

from __future__ import annotations

import asyncio
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
from aris.graph.document_ingestion import DocumentIngestor


@celery_app.task(name="apps.worker.tasks.ingest_task.ingest_document")
def ingest_document(job_id: str, document_id: str) -> None:
    """Execute ingest status transitions and ARIS document parsing."""

    try:
        asyncio.get_running_loop()
    except RuntimeError:
        asyncio.run(_ingest_document_async(job_id, document_id))
        return

    with ThreadPoolExecutor(max_workers=1) as executor:
        future = executor.submit(asyncio.run, _ingest_document_async(job_id, document_id))
        future.result()


async def _ingest_document_async(job_id: str, document_id: str) -> None:
    settings = get_settings()
    database.init_database(settings)

    if database.SessionLocal is None:
        msg = "Database session factory is not initialized"
        raise RuntimeError(msg)

    service = DocumentService(settings)
    ingestor = DocumentIngestor()

    async with database.SessionLocal() as session:
        job = await session.scalar(select(AsyncJob).where(AsyncJob.id == UUID(job_id)))
        document = await session.scalar(select(Document).where(Document.id == UUID(document_id)))

        if job is None or document is None:
            await database.dispose_database()
            return

        job.status = "processing"
        job.started_at = datetime.now(UTC)
        document.status = "processing"
        document.error_msg = None
        await session.commit()

        temp_path = None
        used_fallback_object = False
        try:
            temp_path, used_fallback_object = service.download_to_tempfile(
                document.s3_key, document.filename
            )
            ingest_format = document.file_format if not used_fallback_object else "text"
            ingested_document = ingestor.ingest(temp_path, ingest_format)
            preview = ingested_document.content[:12000]
            document.status = "ready"
            document.metadata_json = {
                **document.metadata_json,
                **{k: v for k, v in ingested_document.metadata.items()},
                "source": ingested_document.source,
                "content_preview": preview,
                "content_length": len(ingested_document.content),
            }
            job.status = "ready"
            job.result = {
                "document_id": str(document.id),
                "status": "ready",
                "ingested_document_id": str(ingested_document.document_id),
            }
            job.error = None
        except Exception as exc:
            if used_fallback_object:
                document.status = "ready"
                document.metadata_json = {
                    **document.metadata_json,
                    "ingest_warning": str(exc),
                    "source": "fallback",
                }
                job.status = "ready"
                job.error = None
                job.result = {"document_id": str(document.id), "status": "ready"}
            else:
                document.status = "failed"
                document.error_msg = str(exc)
                job.status = "failed"
                job.error = str(exc)
                job.result = {"document_id": str(document.id), "status": "failed"}
        finally:
            if temp_path is not None:
                temp_path.unlink(missing_ok=True)
            job.completed_at = datetime.now(UTC)
            await session.commit()

    await database.dispose_database()
