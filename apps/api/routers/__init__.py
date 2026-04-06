"""FastAPI routers for ARIS API."""

from apps.api.routers.auth import router as auth_router
from apps.api.routers.documents import router as documents_router
from apps.api.routers.edges import router as edges_router
from apps.api.routers.graphs import router as graphs_router
from apps.api.routers.jobs import router as jobs_router
from apps.api.routers.plans import router as plans_router
from apps.api.routers.workspaces import router as workspaces_router

__all__ = [
	"auth_router",
	"workspaces_router",
	"documents_router",
	"jobs_router",
	"graphs_router",
	"edges_router",
	"plans_router",
]
