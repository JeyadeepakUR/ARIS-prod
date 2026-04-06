"""Local object storage endpoints used when external S3 is unavailable."""

from __future__ import annotations

from pathlib import Path
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, Request, Response, status
from fastapi.responses import FileResponse
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from apps.api.config import Settings
from apps.api.dependencies import get_config, get_db
from apps.api.models.document import Document
from apps.api.models.job import AsyncJob

router = APIRouter(prefix="/object-storage", tags=["object-storage"])


def _queue_ingest_job(job_id: UUID, document_id: UUID, settings: Settings) -> tuple[bool, str | None]:
    try:
        from apps.worker.tasks.ingest_task import ingest_document

        if settings.celery_task_always_eager:
            ingest_document(str(job_id), str(document_id))
            return True, None

        async_result = ingest_document.delay(str(job_id), str(document_id))
        return True, str(async_result.id)
    except Exception as exc:
        if settings.celery_task_always_eager:
            raise
        return False, str(exc)


def _object_root(settings: Settings) -> Path:
    root = Path(settings.local_object_store_dir)
    root.mkdir(parents=True, exist_ok=True)
    return root


def _resolve_object_path(settings: Settings, bucket: str, object_key: str) -> Path:
    candidate = (_object_root(settings) / bucket / object_key).resolve()
    root = _object_root(settings).resolve()
    if root not in candidate.parents and candidate != root:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Invalid object key")
    return candidate


@router.put("/{bucket}/{object_key:path}", status_code=status.HTTP_204_NO_CONTENT)
async def put_object(
    bucket: str,
    object_key: str,
    request: Request,
    settings: Settings = Depends(get_config),
    session: AsyncSession = Depends(get_db),
) -> Response:
    object_path = _resolve_object_path(settings, bucket, object_key)
    object_path.parent.mkdir(parents=True, exist_ok=True)

    payload = await request.body()
    object_path.write_bytes(payload)

    try:
        segments = object_key.split("/")
        if len(segments) >= 4 and segments[0] == "workspaces" and segments[2] == "documents":
            workspace_id = UUID(segments[1])
            document_id = UUID(segments[3])

            document = await session.scalar(
                select(Document).where(
                    Document.id == document_id,
                    Document.workspace_id == workspace_id,
                )
            )

            if document is not None:
                pending_jobs = await session.scalars(
                    select(AsyncJob)
                    .where(
                        AsyncJob.workspace_id == workspace_id,
                        AsyncJob.job_type == "document_ingest",
                        AsyncJob.status == "pending",
                    )
                    .order_by(AsyncJob.created_at.desc())
                )

                selected_job: AsyncJob | None = None
                for job in pending_jobs:
                    payload_doc_id = str((job.payload or {}).get("document_id", ""))
                    if payload_doc_id == str(document_id):
                        selected_job = job
                        break

                if selected_job is not None:
                    queued, queue_error = _queue_ingest_job(selected_job.id, document.id, settings)
                    if queue_error is not None:
                        selected_job.error = queue_error
                    if queued and not settings.celery_task_always_eager:
                        selected_job.status = "pending"
                    await session.commit()
    except Exception:
        # Keep object upload successful even if async queueing fails.
        pass

    return Response(status_code=status.HTTP_204_NO_CONTENT)


@router.get("/{bucket}/{object_key:path}")
async def get_object(
    bucket: str,
    object_key: str,
    settings: Settings = Depends(get_config),
) -> FileResponse:
    object_path = _resolve_object_path(settings, bucket, object_key)
    if not object_path.exists() or not object_path.is_file():
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Object not found")

    return FileResponse(path=object_path)
