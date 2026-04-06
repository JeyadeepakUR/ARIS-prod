"""FastAPI app factory for ARIS platform API."""

from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy import text

from apps.api.config import get_settings
from apps.api.core.database import Base, dispose_database, get_engine, init_database
from apps.api.routers.auth import router as auth_router
from apps.api.routers.documents import router as documents_router
from apps.api.routers.edges import router as edges_router
from apps.api.routers.graphs import router as graphs_router
from apps.api.routers.hypotheses import router as hypotheses_router
from apps.api.routers.jobs import router as jobs_router
from apps.api.routers.nodes import router as nodes_router
from apps.api.routers.object_storage import router as object_storage_router
from apps.api.routers.plans import router as plans_router
from apps.api.routers.workspaces import router as workspaces_router


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Initialize async DB engine on startup and dispose on shutdown."""

    settings = get_settings()
    init_database(settings)
    current_engine = get_engine()

    if current_engine is not None:
        async with current_engine.begin() as conn:
            await conn.run_sync(Base.metadata.create_all)
            await conn.execute(text("SELECT 1"))

    yield

    await dispose_database()


def create_app() -> FastAPI:
    """Create and configure FastAPI application."""

    settings = get_settings()
    app = FastAPI(title=settings.app_name, version=settings.app_version, lifespan=lifespan)
    allowed_origins = [origin.strip() for origin in settings.cors_allow_origins.split(",") if origin.strip()]
    app.add_middleware(
        CORSMiddleware,
        allow_origins=allowed_origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    app.include_router(auth_router)
    app.include_router(workspaces_router)
    app.include_router(documents_router)
    app.include_router(jobs_router)
    app.include_router(graphs_router)
    app.include_router(nodes_router)
    app.include_router(edges_router)
    app.include_router(hypotheses_router)
    app.include_router(plans_router)
    app.include_router(object_storage_router)

    @app.get("/health")
    async def health() -> dict[str, str]:
        return {"status": "ok"}

    return app


app = create_app()
