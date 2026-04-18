"""Document upload initialization routes."""

from __future__ import annotations

from uuid import UUID, uuid4

from fastapi import APIRouter, Depends, HTTPException, Query
from fastapi import status as http_status
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from apps.api.config import Settings
from apps.api.dependencies import get_config, get_current_user, get_db
from apps.api.models.document import Document
from apps.api.models.job import AsyncJob
from apps.api.models.user import User
from apps.api.models.workspace import Workspace
from apps.api.schemas.document import DocumentRead, DocumentUploadInitRequest, DocumentUploadInitResponse
from apps.api.services.document_service import DocumentService

router = APIRouter(prefix="/workspaces/{workspace_id}/documents", tags=["documents"])


def _document_read(document: Document) -> DocumentRead:
    return DocumentRead(
        id=document.id,
        workspace_id=document.workspace_id,
        filename=document.filename,
        s3_key=document.s3_key,
        file_format=document.file_format,
        file_size_bytes=document.file_size_bytes,
        status=document.status,
        error_msg=document.error_msg,
        metadata=document.metadata_json,
        created_at=document.created_at,
    )


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


@router.post("/upload-url", response_model=DocumentUploadInitResponse, status_code=http_status.HTTP_201_CREATED)
async def initialize_document_upload(
    workspace_id: UUID,
    payload: DocumentUploadInitRequest,
    session: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
    settings: Settings = Depends(get_config),
) -> DocumentUploadInitResponse:
    workspace = await session.scalar(
        select(Workspace).where(Workspace.id == workspace_id, Workspace.owner_id == current_user.id)
    )
    if workspace is None:
        raise HTTPException(status_code=http_status.HTTP_404_NOT_FOUND, detail="Workspace not found")

    document_id = uuid4()
    storage = DocumentService(settings)
    s3_key = storage.build_s3_key(str(workspace.id), str(document_id), payload.filename)

    document = Document(
        id=document_id,
        workspace_id=workspace.id,
        filename=payload.filename,
        s3_key=s3_key,
        file_format=payload.file_format.lower(),
        file_size_bytes=payload.file_size_bytes,
        status="pending",
    )
    session.add(document)
    await session.flush()

    job = AsyncJob(
        workspace_id=workspace.id,
        job_type="document_ingest",
        status="pending",
        payload={"document_id": str(document.id), "s3_key": s3_key},
    )
    session.add(job)
    await session.flush()

    # Persist records before queueing so worker sessions can see them.
    await session.commit()
    await session.refresh(document)
    await session.refresh(job)

    return DocumentUploadInitResponse(
        document=_document_read(document),
        job_id=job.id,
        status=job.status,
        upload_url=storage.generate_upload_url(s3_key),
        queued=False,
    )


@router.get("", response_model=list[DocumentRead])
async def list_documents(
    workspace_id: UUID,
    document_status: str | None = Query(None, alias="status"),
    page: int = Query(1, ge=1),
    size: int = Query(50, ge=1, le=200),
    session: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
) -> list[DocumentRead]:
    workspace = await session.scalar(
        select(Workspace).where(Workspace.id == workspace_id, Workspace.owner_id == current_user.id)
    )
    if workspace is None:
        raise HTTPException(status_code=http_status.HTTP_404_NOT_FOUND, detail="Workspace not found")

    offset = (page - 1) * size
    query = select(Document).where(Document.workspace_id == workspace.id)
    if document_status:
        query = query.where(Document.status == document_status)
    query = query.order_by(Document.created_at.desc()).offset(offset).limit(size)

    rows = await session.scalars(query)
    return [_document_read(document) for document in rows]
