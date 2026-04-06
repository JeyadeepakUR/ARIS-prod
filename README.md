# ARIS

Autonomous Research Intelligence System (ARIS) is an evidence-first research intelligence platform.

ARIS ingests documents, builds typed knowledge graphs, and surfaces research actions with traceable evidence for each graph edge.

## Highlights

- Evidence-backed graph generation (`belongs_to_domain`, `has_concept`, `cross_domain_bridge`)
- Async ingestion and graph build pipeline (API + worker)
- Domain-network strategy for interpretable cross-domain linking
- FastAPI backend and Next.js dashboard
- Local development support (SQLite + eager tasks) and production-ready architecture

## Monorepo Layout

```text
apps/
  api/       FastAPI service (auth, workspaces, documents, graphs, jobs)
  web/       Next.js dashboard and graph canvas UI
  worker/    Celery tasks for ingest/build workflows
aris/        Core deterministic engine modules
packages/    Packaged variants / SDK workspaces
infra/       Deployment and infrastructure assets
```

## Prerequisites

- Python 3.11+
- Node.js 20+
- (Optional) Redis for non-eager background jobs
- (Optional) PostgreSQL for production-like local runs

## Quick Start (Local)

From repository root:

```powershell
python -m venv .venv
& .\.venv\Scripts\Activate.ps1
pip install -e .
```

### Run API (SQLite + eager worker mode)

```powershell
$env:DATABASE_URL="sqlite+aiosqlite:///db.sqlite3"
$env:CELERY_TASK_ALWAYS_EAGER="true"
$env:REDIS_URL="redis://localhost:6379/0"
python -m uvicorn apps.api.main:app --host 127.0.0.1 --port 8000 --reload
```

API docs: `http://127.0.0.1:8000/docs`

### Run Web

```powershell
cd apps/web
npm install
npm run dev
```

Web app: `http://localhost:3000`

## Common Startup Issue

If startup fails with `socket.gaierror: [Errno 11001] getaddrinfo failed`, your `DATABASE_URL` is pointing to a hostname (for example `postgres`) that is resolvable only inside Docker networks.

Use SQLite locally:

```powershell
$env:DATABASE_URL="sqlite+aiosqlite:///db.sqlite3"
```

Or use a valid reachable PostgreSQL host.

## Testing

```powershell
python -m pytest tests -q
```

API tests:

```powershell
python -m pytest apps/api/tests -q
```

Web lint:

```powershell
cd apps/web
npm run lint
```

## Product Vision

The full product vision and long-range architecture are documented in `product_architecture.md`.

## License

This repository currently has no license file attached.