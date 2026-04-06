"""Celery worker bootstrap for async ingest jobs."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from celery import Celery

from apps.api.config import get_settings


def create_celery_app() -> Any:
    """Create configured Celery app from API settings."""

    from celery import Celery

    settings = get_settings()
    broker_url = settings.celery_broker_url or settings.redis_url
    result_backend = settings.celery_result_backend or settings.redis_url

    celery_app = Celery(
        "aris_worker",
        broker=broker_url,
        backend=result_backend,
        include=[
            "apps.worker.tasks.ingest_task",
            "apps.worker.tasks.graph_build_task",
            "apps.worker.tasks.plan_task",
        ],
    )
    celery_app.conf.update(task_serializer="json", result_serializer="json", accept_content=["json"])
    celery_app.conf.task_always_eager = settings.celery_task_always_eager
    return celery_app


celery_app = create_celery_app()


if __name__ == "__main__":
    celery_app.worker_main(["worker", "--loglevel=info"])
