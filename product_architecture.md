# ARIS — STARTUP PRODUCT ARCHITECTURE SPEC
### Autonomous Research Intelligence System | Production Launch Blueprint
**Version:** 1.0 | **Author:** Jeyadeepak U R | **Date:** April 2026  
**Purpose:** Complete coding-agent-ready specification for transforming ARIS from a Python library into a launchable SaaS product.

---

## 0. NORTH STAR & PRODUCT POSITIONING

**What ARIS sells:** "Upload your documents. ARIS builds a living knowledge graph, finds connections you'd miss, and tells you exactly what to research next — with evidence for every claim it makes."

**Who it's for (Tier 1 launch targets):**
- PhD researchers / academics managing 50–500 paper corpora
- Independent analysts and consultants synthesising reports
- Enterprise R&D teams connecting internal knowledge bases
- Legal professionals linking cases, statutes, precedents

**Core moat:** Every other AI research tool is a black box. ARIS is the only one that can *prove* every connection it draws — because Phase 1's evidence-complete edge system is the product itself.

**Monetisation model:**
- Free tier: 1 workspace, 25 documents, 1 knowledge graph
- Pro ($19/mo): 5 workspaces, 500 documents, unlimited graphs, API access
- Team ($79/mo per 5 seats): Shared workspaces, collaborative annotations
- Enterprise (custom): On-prem deployment, SSO, SAML, SLA

---

## 1. MONOREPO STRUCTURE (TOP LEVEL)

```
aris-platform/
├── apps/
│   ├── web/                    # Next.js 14 frontend (user dashboard + graph UI)
│   ├── api/                    # FastAPI backend (Python 3.11)
│   └── worker/                 # Celery async task workers
│
├── packages/
│   ├── aris-core/              # YOUR EXISTING aris-prod repo (Git subtree)
│   └── aris-sdk/               # Public Python SDK (thin client over API)
│
├── infra/
│   ├── docker/                 # Dockerfiles per service
│   ├── k8s/                    # Kubernetes manifests (production)
│   ├── terraform/              # Cloud infra-as-code (AWS/GCP)
│   └── nginx/                  # Reverse proxy config
│
├── scripts/
│   ├── seed_dev.py             # Populate dev DB with test data
│   ├── migrate.sh              # Run Alembic migrations
│   └── smoke_test.sh           # Post-deploy health checks
│
├── .github/
│   └── workflows/
│       ├── ci.yml              # Test + lint on every PR
│       ├── deploy-staging.yml  # Auto-deploy to staging on main merge
│       └── deploy-prod.yml     # Manual-gate prod deploy
│
├── docker-compose.yml          # Full local dev stack
├── docker-compose.test.yml     # Isolated test environment
├── .env.example                # All required env vars documented
└── Makefile                    # Developer shortcuts (make dev, make test, etc.)
```

---

## 2. FULL DIRECTORY TREE WITH FILE SPECIFICATIONS

### 2.1 `apps/api/` — FastAPI Backend

```
apps/api/
├── main.py                     # FastAPI app factory, lifespan, middleware registration
├── config.py                   # Pydantic Settings — loads from .env, typed config object
├── dependencies.py             # FastAPI Depends() factories (db session, current user, etc.)
│
├── routers/
│   ├── __init__.py
│   ├── auth.py                 # POST /auth/register, /auth/login, /auth/refresh, /auth/logout
│   ├── workspaces.py           # CRUD /workspaces — tenant isolation root
│   ├── documents.py            # POST /workspaces/{id}/documents (upload), GET/DELETE
│   ├── graphs.py               # POST /workspaces/{id}/graphs (build), GET graph+edges
│   ├── edges.py                # GET /edges/{id} (full evidence), PATCH (annotate)
│   ├── plans.py                # POST /graphs/{id}/plan, GET planned actions
│   ├── jobs.py                 # GET /jobs/{id} — async job status polling
│   └── billing.py              # Stripe webhooks, subscription status
│
├── models/
│   ├── __init__.py
│   ├── user.py                 # SQLAlchemy: User (id, email, hashed_pw, plan, created_at)
│   ├── workspace.py            # SQLAlchemy: Workspace (id, owner_id, name, slug, settings)
│   ├── membership.py           # SQLAlchemy: WorkspaceMember (workspace_id, user_id, role)
│   ├── document.py             # SQLAlchemy: Document (id, workspace_id, filename, s3_key, status)
│   ├── graph.py                # SQLAlchemy: KnowledgeGraph (id, workspace_id, node_count, edge_count)
│   ├── node.py                 # SQLAlchemy: GraphNode (id, graph_id, label, node_type, metadata_json)
│   ├── edge.py                 # SQLAlchemy: GraphEdge (id, graph_id, src, tgt, type, confidence, evidence_json)
│   ├── plan_action.py          # SQLAlchemy: ResearchAction (id, graph_id, type, priority, rationale)
│   └── job.py                  # SQLAlchemy: AsyncJob (id, type, status, payload_json, result_json)
│
├── schemas/
│   ├── __init__.py
│   ├── auth.py                 # Pydantic: RegisterRequest, LoginRequest, TokenResponse
│   ├── workspace.py            # Pydantic: WorkspaceCreate, WorkspaceRead, WorkspaceUpdate
│   ├── document.py             # Pydantic: DocumentRead, DocumentStatus
│   ├── graph.py                # Pydantic: GraphRead, GraphSummary, GraphBuildRequest
│   ├── edge.py                 # Pydantic: EdgeRead, EdgeDetail (with full evidence chain)
│   ├── plan.py                 # Pydantic: PlanRequest, ResearchActionRead
│   └── job.py                  # Pydantic: JobRead, JobStatus enum
│
├── services/
│   ├── __init__.py
│   ├── auth_service.py         # bcrypt hashing, JWT issue/verify, refresh token rotation
│   ├── document_service.py     # S3 upload, virus scan trigger, metadata extraction
│   ├── graph_service.py        # Orchestrates aris-core: ingest → link → materialise
│   ├── plan_service.py         # Calls aris-core ResearchPlanner, persists actions
│   ├── job_service.py          # Enqueues Celery tasks, polls status, handles failures
│   ├── quota_service.py        # Checks plan limits before expensive operations
│   └── billing_service.py      # Stripe customer/subscription create, webhook handling
│
├── core/
│   ├── __init__.py
│   ├── security.py             # JWT secret, algorithm, token expiry constants
│   ├── exceptions.py           # Custom HTTPException subclasses (QuotaExceeded, NotFound, etc.)
│   ├── middleware.py           # RateLimitMiddleware, RequestIDMiddleware, CORSMiddleware
│   ├── logging.py              # Structured JSON logging (correlation IDs)
│   └── database.py             # SQLAlchemy async engine, session factory, Base
│
├── tasks/                      # Celery task definitions (imported by worker)
│   ├── __init__.py
│   ├── ingest_task.py          # Task: download from S3, run DocumentIngestor, update DB
│   ├── graph_build_task.py     # Task: run full aris-core graph pipeline, persist nodes+edges
│   └── plan_task.py            # Task: run ResearchPlanner on completed graph
│
├── migrations/
│   ├── env.py                  # Alembic env (async SQLAlchemy)
│   ├── script.py.mako          # Migration template
│   └── versions/               # Auto-generated migration files
│
├── tests/
│   ├── conftest.py             # pytest fixtures: test db, test client, mock S3
│   ├── test_auth.py
│   ├── test_documents.py
│   ├── test_graphs.py
│   ├── test_edges.py
│   └── test_jobs.py
│
├── pyproject.toml              # API-specific deps (fastapi, sqlalchemy, celery, boto3, stripe)
└── Dockerfile                  # Multi-stage: builder → slim runtime image
```

---

### 2.2 `apps/worker/` — Celery Async Worker

```
apps/worker/
├── main.py                     # Celery app init, broker URL, result backend config
├── config.py                   # Worker-specific settings (concurrency, queues, retry policy)
│
├── queues/
│   ├── ingest.py               # Queue: document.ingest — downloads, extracts, updates status
│   ├── graph.py                # Queue: graph.build — full ARIS pipeline, can take 10–120s
│   └── plan.py                 # Queue: plan.generate — fast (< 5s), low priority queue
│
├── beat_schedule.py            # Celery Beat: periodic jobs (cleanup, usage aggregation)
├── Dockerfile
└── pyproject.toml
```

---

### 2.3 `apps/web/` — Next.js 14 Frontend

```
apps/web/
├── app/                        # Next.js App Router
│   ├── layout.tsx              # Root layout — providers (auth, theme, query)
│   ├── page.tsx                # Landing page (marketing)
│   ├── (auth)/
│   │   ├── login/page.tsx
│   │   └── register/page.tsx
│   └── (dashboard)/
│       ├── layout.tsx          # Dashboard shell: sidebar + topbar
│       ├── page.tsx            # Dashboard home: recent workspaces, usage stats
│       ├── workspaces/
│       │   ├── page.tsx        # Workspace list
│       │   └── [id]/
│       │       ├── page.tsx    # Workspace overview
│       │       ├── documents/
│       │       │   └── page.tsx    # Document list + upload dropzone
│       │       ├── graphs/
│       │       │   ├── page.tsx    # Graph list
│       │       │   └── [graphId]/
│       │       │       ├── page.tsx        # Graph canvas (React Flow)
│       │       │       └── edge/[edgeId]/page.tsx  # Edge evidence deep-dive
│       │       └── plans/
│       │           └── page.tsx    # Research action board (Kanban style)
│       └── settings/
│           ├── page.tsx        # Account settings
│           └── billing/page.tsx    # Plan upgrade, usage meters
│
├── components/
│   ├── ui/                     # shadcn/ui primitives (Button, Dialog, Toast, etc.)
│   ├── layout/
│   │   ├── Sidebar.tsx
│   │   ├── Topbar.tsx
│   │   └── PageHeader.tsx
│   ├── graph/
│   │   ├── GraphCanvas.tsx     # React Flow wrapper — nodes, edges, controls
│   │   ├── NodeCard.tsx        # Custom React Flow node: label, type badge, document link
│   │   ├── EdgeTooltip.tsx     # Hover tooltip: confidence score, evidence preview
│   │   └── GraphControls.tsx   # Zoom, fit, filter by confidence/type controls
│   ├── documents/
│   │   ├── UploadDropzone.tsx  # react-dropzone: PDF + TXT, size/count quota check
│   │   ├── DocumentCard.tsx    # Status pill (processing/ready/error), metadata
│   │   └── DocumentList.tsx
│   ├── plans/
│   │   ├── ActionBoard.tsx     # Kanban: Gap / Contradiction / Weak-Evidence columns
│   │   └── ActionCard.tsx      # Priority badge, rationale text, evidence nodes
│   └── shared/
│       ├── JobStatusBadge.tsx  # Live polling badge: queued → processing → done
│       ├── ConfidenceBar.tsx   # Visual 0–1 confidence score bar
│       └── EvidenceChain.tsx   # Expandable reasoning steps accordion
│
├── lib/
│   ├── api/
│   │   ├── client.ts           # Axios instance with auth interceptors, base URL
│   │   ├── workspaces.ts       # API call functions for workspace endpoints
│   │   ├── documents.ts
│   │   ├── graphs.ts
│   │   ├── edges.ts
│   │   └── jobs.ts
│   ├── hooks/
│   │   ├── useWorkspace.ts     # SWR/React Query: workspace data
│   │   ├── useGraph.ts         # Graph data + polling while job running
│   │   ├── useJobPoller.ts     # Polls /jobs/{id} every 2s until terminal state
│   │   └── useAuth.ts          # Auth state, login/logout actions
│   ├── stores/
│   │   └── graphStore.ts       # Zustand: selected node/edge, filter state, layout mode
│   └── utils/
│       ├── confidence.ts       # Confidence score → colour mapping
│       └── truncate.ts         # Text truncation helpers
│
├── public/
│   └── logo.svg
│
├── next.config.ts
├── tailwind.config.ts
├── tsconfig.json
├── package.json
└── Dockerfile
```

---

### 2.4 `packages/aris-core/` — Your Existing Library (Promoted)

```
packages/aris-core/             # Git subtree from JeyadeepakUR/ARIS-prod
├── aris/
│   ├── core/                   # Modules 0–6 (SEALED, no changes)
│   ├── graph/                  # Modules 7–9 (SEALED, no changes)
│   └── ml/                     # Modules 10–13 (Phase 2, pending)
├── tests/
├── PHASE_1_CONTRACT.md
└── pyproject.toml
```

**CRITICAL RULE FOR CODING AGENT:** Never modify any file inside `packages/aris-core/aris/core/` or `packages/aris-core/aris/graph/`. These are sealed. The API layer in `apps/api/services/graph_service.py` wraps them — all product logic lives in the service layer, not in the core library.

---

### 2.5 `packages/aris-sdk/` — Public Python SDK

```
packages/aris-sdk/
├── aris_sdk/
│   ├── __init__.py             # Exports: ARISClient
│   ├── client.py               # ARISClient: init(api_key), all methods
│   ├── models.py               # Pydantic response models (mirrors API schemas)
│   └── exceptions.py           # ARISError, QuotaError, AuthError
├── examples/
│   ├── upload_and_build.py     # End-to-end example
│   └── query_graph.py
├── README.md
└── pyproject.toml
```

---

### 2.6 `infra/` — Infrastructure

```
infra/
├── docker/
│   ├── api.Dockerfile          # Multi-stage: python:3.11-slim, install deps, copy app
│   ├── worker.Dockerfile       # Same base as api, different CMD
│   └── web.Dockerfile          # node:20-alpine, build Next.js, serve with standalone output
│
├── k8s/
│   ├── namespace.yaml
│   ├── api/
│   │   ├── deployment.yaml     # 2 replicas min, resource limits, health probes
│   │   ├── service.yaml        # ClusterIP
│   │   └── hpa.yaml            # HPA: scale on CPU 60%
│   ├── worker/
│   │   ├── deployment.yaml     # 1–4 replicas, scale on queue depth (KEDA)
│   │   └── keda-scaler.yaml    # KEDA ScaledObject watching Redis queue length
│   ├── web/
│   │   ├── deployment.yaml
│   │   └── service.yaml
│   └── ingress.yaml            # nginx-ingress: app.aris.ai → web, api.aris.ai → api
│
└── terraform/
    ├── main.tf                 # Provider config (AWS)
    ├── variables.tf
    ├── outputs.tf
    ├── modules/
    │   ├── rds/                # PostgreSQL 15 RDS (multi-AZ for prod)
    │   ├── elasticache/        # Redis 7 (Celery broker + result backend)
    │   ├── s3/                 # Document storage bucket (server-side encryption)
    │   ├── eks/                # EKS cluster (production)
    │   └── cloudfront/         # CDN for Next.js static assets
    └── environments/
        ├── dev.tfvars
        ├── staging.tfvars
        └── prod.tfvars
```

---

## 3. TECHNOLOGY STACK — DEFINITIVE CHOICES

| Layer | Technology | Reason |
|---|---|---|
| **Core Engine** | Python 3.11, aris-core (sealed) | Your existing moat — don't change it |
| **API Framework** | FastAPI + Pydantic v2 | Async, auto OpenAPI docs, type-safe |
| **ORM** | SQLAlchemy 2.0 (async) + Alembic | Production-grade, async-native |
| **Database** | PostgreSQL 15 | JSONB for evidence/metadata, battle-tested |
| **Task Queue** | Celery 5 + Redis 7 | Handles 10–120s ARIS pipeline jobs reliably |
| **File Storage** | AWS S3 (boto3) | Pre-signed upload URLs, lifecycle policies |
| **Auth** | JWT (python-jose) + bcrypt | Stateless, refresh token rotation |
| **Frontend** | Next.js 14 (App Router) + TypeScript | SSR for landing, CSR for dashboard |
| **Graph UI** | React Flow | Best-in-class node/edge canvas |
| **State** | Zustand + React Query (TanStack) | Lightweight + server state caching |
| **Styling** | Tailwind CSS + shadcn/ui | Fast, consistent, accessible |
| **Payments** | Stripe | Subscriptions + usage-based billing |
| **Monitoring** | Sentry (errors) + Prometheus/Grafana (metrics) | Full observability |
| **Logging** | structlog (JSON) + Datadog or Grafana Loki | Correlation IDs on every request |
| **CI/CD** | GitHub Actions | Native to your repo |
| **Container** | Docker + Kubernetes (EKS) | Scale worker pods on queue depth |
| **IaC** | Terraform | Reproducible infra |

---

## 4. DATABASE SCHEMA

```sql
-- Users & Auth
CREATE TABLE users (
    id          UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    email       VARCHAR(255) UNIQUE NOT NULL,
    hashed_pw   VARCHAR(255) NOT NULL,
    plan        VARCHAR(50) NOT NULL DEFAULT 'free',  -- free | pro | team | enterprise
    stripe_customer_id VARCHAR(255),
    is_verified BOOLEAN DEFAULT FALSE,
    created_at  TIMESTAMPTZ DEFAULT NOW(),
    updated_at  TIMESTAMPTZ DEFAULT NOW()
);

CREATE TABLE refresh_tokens (
    id          UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id     UUID NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    token_hash  VARCHAR(255) UNIQUE NOT NULL,
    expires_at  TIMESTAMPTZ NOT NULL,
    created_at  TIMESTAMPTZ DEFAULT NOW()
);

-- Workspaces (Multi-tenant isolation unit)
CREATE TABLE workspaces (
    id          UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    owner_id    UUID NOT NULL REFERENCES users(id),
    name        VARCHAR(255) NOT NULL,
    slug        VARCHAR(255) UNIQUE NOT NULL,
    settings    JSONB DEFAULT '{}',
    created_at  TIMESTAMPTZ DEFAULT NOW()
);

CREATE TABLE workspace_members (
    workspace_id UUID REFERENCES workspaces(id) ON DELETE CASCADE,
    user_id      UUID REFERENCES users(id) ON DELETE CASCADE,
    role         VARCHAR(50) NOT NULL DEFAULT 'viewer',  -- owner | editor | viewer
    PRIMARY KEY (workspace_id, user_id)
);

-- Documents
CREATE TABLE documents (
    id           UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    workspace_id UUID NOT NULL REFERENCES workspaces(id) ON DELETE CASCADE,
    filename     VARCHAR(500) NOT NULL,
    s3_key       VARCHAR(1000) NOT NULL,
    file_format  VARCHAR(50) NOT NULL,  -- txt | pdf
    file_size_bytes BIGINT,
    status       VARCHAR(50) NOT NULL DEFAULT 'pending',  -- pending | processing | ready | error
    error_msg    TEXT,
    metadata     JSONB DEFAULT '{}',
    created_at   TIMESTAMPTZ DEFAULT NOW()
);

-- Knowledge Graphs
CREATE TABLE knowledge_graphs (
    id           UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    workspace_id UUID NOT NULL REFERENCES workspaces(id) ON DELETE CASCADE,
    name         VARCHAR(255) NOT NULL,
    status       VARCHAR(50) NOT NULL DEFAULT 'pending',
    node_count   INT DEFAULT 0,
    edge_count   INT DEFAULT 0,
    config       JSONB DEFAULT '{}',  -- linking strategies, confidence threshold
    created_at   TIMESTAMPTZ DEFAULT NOW(),
    completed_at TIMESTAMPTZ
);

CREATE TABLE graph_nodes (
    id         UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    graph_id   UUID NOT NULL REFERENCES knowledge_graphs(id) ON DELETE CASCADE,
    document_id UUID REFERENCES documents(id),
    label      VARCHAR(500) NOT NULL,
    node_type  VARCHAR(100),
    metadata   JSONB DEFAULT '{}'
);

CREATE TABLE graph_edges (
    id                  UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    graph_id            UUID NOT NULL REFERENCES knowledge_graphs(id) ON DELETE CASCADE,
    source_node_id      UUID NOT NULL REFERENCES graph_nodes(id),
    target_node_id      UUID NOT NULL REFERENCES graph_nodes(id),
    edge_type           VARCHAR(100) NOT NULL,
    confidence          FLOAT NOT NULL CHECK (confidence >= 0 AND confidence <= 1),
    evidence            JSONB NOT NULL,  -- reasoning_steps array from ARIS core
    reasoning_trace_id  UUID NOT NULL,  -- links to aris_traces table
    metadata            JSONB DEFAULT '{}'
);

-- ARIS Core Traces (full audit log)
CREATE TABLE aris_traces (
    id         UUID PRIMARY KEY,  -- = request_id from InputPacket
    graph_id   UUID REFERENCES knowledge_graphs(id),
    trace_json JSONB NOT NULL,    -- full MemoryTrace serialised
    created_at TIMESTAMPTZ DEFAULT NOW()
);

-- Research Plans
CREATE TABLE research_actions (
    id          UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    graph_id    UUID NOT NULL REFERENCES knowledge_graphs(id) ON DELETE CASCADE,
    action_type VARCHAR(100) NOT NULL,  -- investigate_gap | resolve_contradiction | strengthen_evidence
    priority    FLOAT NOT NULL,
    rationale   TEXT NOT NULL,
    evidence    JSONB NOT NULL,   -- node_ids or edge_ids referenced
    status      VARCHAR(50) DEFAULT 'proposed',  -- proposed | in_progress | done | dismissed
    created_at  TIMESTAMPTZ DEFAULT NOW()
);

-- Async Jobs
CREATE TABLE async_jobs (
    id          UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    workspace_id UUID REFERENCES workspaces(id),
    job_type    VARCHAR(100) NOT NULL,  -- document_ingest | graph_build | plan_generate
    status      VARCHAR(50) NOT NULL DEFAULT 'queued',  -- queued | running | done | failed
    celery_task_id VARCHAR(255),
    payload     JSONB DEFAULT '{}',
    result      JSONB DEFAULT '{}',
    error       TEXT,
    created_at  TIMESTAMPTZ DEFAULT NOW(),
    started_at  TIMESTAMPTZ,
    completed_at TIMESTAMPTZ
);

-- Indexes
CREATE INDEX idx_documents_workspace ON documents(workspace_id);
CREATE INDEX idx_documents_status ON documents(status);
CREATE INDEX idx_graph_nodes_graph ON graph_nodes(graph_id);
CREATE INDEX idx_graph_edges_graph ON graph_edges(graph_id);
CREATE INDEX idx_graph_edges_confidence ON graph_edges(confidence);
CREATE INDEX idx_research_actions_graph ON research_actions(graph_id);
CREATE INDEX idx_async_jobs_workspace_status ON async_jobs(workspace_id, status);
```

---

## 5. API ROUTES — COMPLETE SPECIFICATION

### 5.1 Authentication
```
POST   /auth/register          Body: {email, password}            → {user, access_token, refresh_token}
POST   /auth/login             Body: {email, password}            → {access_token, refresh_token}
POST   /auth/refresh           Body: {refresh_token}              → {access_token}
POST   /auth/logout            Auth: Bearer                       → 204
GET    /auth/me                Auth: Bearer                       → User
```

### 5.2 Workspaces
```
GET    /workspaces             Auth: Bearer                       → Workspace[]
POST   /workspaces             Body: {name}                       → Workspace
GET    /workspaces/{id}        Auth: Bearer                       → Workspace
PATCH  /workspaces/{id}        Body: {name, settings}             → Workspace
DELETE /workspaces/{id}        Auth: Bearer                       → 204
POST   /workspaces/{id}/members  Body: {email, role}              → Member
DELETE /workspaces/{id}/members/{userId}                          → 204
```

### 5.3 Documents
```
GET    /workspaces/{id}/documents                                 → Document[]
POST   /workspaces/{id}/documents  Body: multipart/form-data     → {document, job_id}
GET    /workspaces/{id}/documents/{docId}                         → Document
DELETE /workspaces/{id}/documents/{docId}                         → 204
```

### 5.4 Knowledge Graphs
```
GET    /workspaces/{id}/graphs                                    → Graph[]
POST   /workspaces/{id}/graphs   Body: {name, document_ids[], config}  → {graph, job_id}
GET    /workspaces/{id}/graphs/{graphId}                          → GraphDetail (nodes+edges summary)
GET    /workspaces/{id}/graphs/{graphId}/nodes                    → Node[]
GET    /workspaces/{id}/graphs/{graphId}/edges                    → Edge[] (paginated, filterable by confidence)
GET    /workspaces/{id}/graphs/{graphId}/edges/{edgeId}           → EdgeDetail (full evidence chain)
DELETE /workspaces/{id}/graphs/{graphId}                          → 204
```

### 5.5 Research Plans
```
POST   /workspaces/{id}/graphs/{graphId}/plan  Body: {strategy, max_actions}  → {actions[], job_id}
GET    /workspaces/{id}/graphs/{graphId}/plan  Auth: Bearer                   → ResearchAction[]
PATCH  /workspaces/{id}/graphs/{graphId}/plan/{actionId}  Body: {status}      → ResearchAction
```

### 5.6 Jobs
```
GET    /jobs/{jobId}           Auth: Bearer                       → Job (status, progress, result)
```

### 5.7 Billing
```
GET    /billing/plans                                             → Plan[] (public)
POST   /billing/subscribe      Body: {plan, payment_method_id}   → Subscription
GET    /billing/usage          Auth: Bearer                       → UsageSummary
POST   /billing/webhook        Stripe-Signature header            → 200
```

---

## 6. SECURITY ARCHITECTURE

### 6.1 Authentication & Authorisation
- **JWT access tokens**: 15-minute expiry, signed with RS256 (asymmetric — separate sign/verify keys)
- **Refresh tokens**: 30-day expiry, stored as bcrypt hash in DB, rotated on every use
- **Workspace RBAC**: Every request to a workspace resource checks `workspace_members` — owner/editor/viewer
- **Row-level isolation**: Every SQL query filters by `workspace_id`, making cross-tenant data leaks structurally impossible
- **API keys** (Pro+): SHA-256 hashed in DB, prefix `aris_` for easy secret scanning detection

### 6.2 Input Validation & Injection Prevention
- All inputs validated by Pydantic v2 before hitting any service layer
- File uploads: MIME type validation + magic bytes check (don't trust Content-Type header alone)
- File size limits enforced before S3 upload (free: 10MB/file, pro: 100MB/file)
- Document quota checked before S3 upload, not after
- No raw SQL — SQLAlchemy ORM only; parameterised queries everywhere

### 6.3 File Storage Security
- Documents stored in private S3 bucket — no public access policy
- Pre-signed upload URLs (PUT, 10-min expiry) issued by API — browser uploads directly to S3
- Pre-signed download URLs (GET, 5-min expiry) issued on demand — never expose S3 keys to client
- S3 bucket: server-side encryption (SSE-S3), versioning enabled, lifecycle: move to Glacier after 90 days
- Virus scanning: ClamAV Lambda triggered on every S3 ObjectCreated event before ingest task runs

### 6.4 Rate Limiting
- Per-IP rate limit: 100 req/min on auth endpoints (nginx level)
- Per-user rate limit: 1000 req/hour on API endpoints (Redis token bucket in middleware)
- Graph build rate: 5 concurrent builds per workspace (enforced in job_service before Celery enqueue)
- Document upload rate: 10 uploads/hour on free tier (quota_service)

### 6.5 Infrastructure Security
- All services communicate over private VPC — API and worker never exposed to internet directly
- Nginx reverse proxy terminates TLS (Let's Encrypt / ACM)
- Secrets in AWS Secrets Manager, injected as env vars at pod startup — never in Docker images or git
- PostgreSQL: SSL enforced, credentials rotated monthly, no public endpoint
- Redis: auth enabled, TLS, no public endpoint
- GitHub Actions: OIDC-based AWS auth (no long-lived AWS keys in CI)
- Dependabot enabled for Python and npm dependencies

### 6.6 ARIS Core Isolation
- `aris-core` runs in the Celery worker process — completely isolated from the HTTP request lifecycle
- Worker has no inbound network access — it can only read from S3 and write to PostgreSQL/Redis
- If the ARIS pipeline crashes (memory error, etc.), Celery retries with exponential backoff (max 3 retries)
- Worker pods have memory limits in Kubernetes — prevents OOM from large corpora taking down the API

---

## 7. ASYNC JOB ARCHITECTURE (CRITICAL FOR UX)

ARIS graph builds can take 10–120 seconds for large corpora. This is handled as follows:

```
User clicks "Build Graph"
       ↓
POST /workspaces/{id}/graphs
       ↓
graph_service.py creates KnowledgeGraph record (status=pending)
       ↓
job_service.py creates AsyncJob record + enqueues Celery task
       ↓
API returns immediately: { graph_id, job_id }    ← < 200ms response
       ↓
Frontend polls GET /jobs/{jobId} every 2s (useJobPoller hook)
       ↓
Worker picks up task:
  1. ingest_task: Download docs from S3 → run DocumentIngestor → store Document objects
  2. graph_build_task: Run aris-core pipeline → persist nodes+edges → update graph status=completed
  3. plan_task: Auto-run ResearchPlanner → persist ResearchActions
       ↓
Job status → "done" → Frontend stops polling → React Query invalidates graph cache
       ↓
Graph canvas renders with React Flow
```

**Job progress events** (optional enhancement, Sprint 3+): Use Redis pub/sub to push progress updates to the frontend via Server-Sent Events (SSE endpoint: `GET /jobs/{jobId}/stream`).

---

## 8. SPRINT PLAN — MAXIMUM VELOCITY BUILD ORDER

This is designed for a solo developer + coding agent. Each sprint = 1 week. Total: 8 weeks to launchable MVP.

### SPRINT 1 — Foundation (Week 1)
**Goal: Everything runs locally in Docker**

| Task | Files to Create | Agent Instructions |
|---|---|---|
| Monorepo scaffold | All top-level dirs, docker-compose.yml, Makefile | Create skeleton, no logic |
| DB schema | `migrations/versions/001_initial.sql` | Full schema from Section 4 above |
| FastAPI app factory | `apps/api/main.py`, `config.py`, `database.py` | Lifespan: connect DB on startup |
| Auth models + schemas | `models/user.py`, `schemas/auth.py` | SQLAlchemy + Pydantic exactly as specced |
| Auth routes | `routers/auth.py`, `services/auth_service.py` | JWT RS256, bcrypt, refresh rotation |
| Docker Compose | `docker-compose.yml` | api + postgres + redis + worker services |

**Sprint 1 done when:** `make dev` starts all services, `POST /auth/register` and `POST /auth/login` work, tokens validate.

---

### SPRINT 2 — Core Data (Week 2)
**Goal: Documents upload and get stored**

| Task | Files to Create |
|---|---|
| Workspace CRUD | `routers/workspaces.py`, `models/workspace.py`, `services/` |
| S3 integration | `services/document_service.py` — pre-signed upload/download URLs |
| Document upload endpoint | `routers/documents.py` |
| Celery worker bootstrap | `apps/worker/main.py`, `tasks/ingest_task.py` |
| Ingest task | Downloads from S3, runs `aris-core` DocumentIngestor, updates DB status |
| Job polling endpoint | `routers/jobs.py` |

**Sprint 2 done when:** Upload a PDF → status goes pending → processing → ready. GET /jobs/{id} returns correct status.

---

### SPRINT 3 — ARIS Pipeline Integration (Week 3)
**Goal: Knowledge graphs build end-to-end**

| Task | Files to Create |
|---|---|
| Graph service | `services/graph_service.py` — wraps aris-core graph pipeline |
| Graph build task | `tasks/graph_build_task.py` — full ingest → link → materialise → plan |
| Persist nodes + edges | `models/node.py`, `models/edge.py`, graph persistence logic |
| Graph routes | `routers/graphs.py`, `routers/edges.py` |
| ARIS trace storage | `models/job.py`, trace JSON serialisation |
| Plan task + routes | `tasks/plan_task.py`, `routers/plans.py` |

**Sprint 3 done when:** POST /workspaces/{id}/graphs with 3 docs → async job completes → GET /graphs/{id}/edges returns edges with evidence JSON.

---

### SPRINT 4 — Frontend Shell (Week 4)
**Goal: Users can log in and see their workspaces**

| Task | Files to Create |
|---|---|
| Next.js project init | `apps/web/` scaffold, Tailwind, shadcn/ui |
| Auth pages | Login, Register with form validation |
| API client | `lib/api/client.ts` with auth interceptors |
| Auth hook | `lib/hooks/useAuth.ts` |
| Dashboard shell | Sidebar, Topbar, layout |
| Workspace list + create | `app/(dashboard)/workspaces/page.tsx` |
| Document upload page | Dropzone, status polling with useJobPoller |

**Sprint 4 done when:** Full auth flow works in browser. Upload a document and watch status update in real-time.

---

### SPRINT 5 — Graph Canvas (Week 5)
**Goal: Knowledge graph renders and is interactive**

| Task | Files to Create |
|---|---|
| React Flow integration | `components/graph/GraphCanvas.tsx` |
| Custom node + edge components | `NodeCard.tsx`, `EdgeTooltip.tsx` |
| Graph build trigger UI | Select documents, config options, trigger build |
| Job progress UI | `JobStatusBadge.tsx`, polling while building |
| Edge evidence panel | Click edge → slide-out panel → full evidence chain |
| Confidence filter | Slider to filter edges below threshold |
| Graph store | `lib/stores/graphStore.ts` |

**Sprint 5 done when:** Build a graph from the UI, watch it appear in React Flow, click an edge and see the full reasoning chain.

---

### SPRINT 6 — Research Plans + Polish (Week 6)
**Goal: Research Action board is live, UI is shippable**

| Task | Files to Create |
|---|---|
| Action board (Kanban) | `components/plans/ActionBoard.tsx`, `ActionCard.tsx` |
| Plan trigger UI | "Generate Research Plan" button on graph page |
| Action status update | PATCH /plan/{actionId} — mark in_progress / done / dismissed |
| Settings page | Account info, password change |
| Error states + empty states | All pages handle loading/error/empty correctly |
| Toast notifications | Global toast for job completion, errors |

**Sprint 6 done when:** End-to-end user story works: register → upload docs → build graph → view evidence → generate plan → manage actions.

---

### SPRINT 7 — Billing + Quotas (Week 7)
**Goal: Users can upgrade, quotas are enforced**

| Task | Files to Create |
|---|---|
| Stripe integration | `services/billing_service.py`, webhook handler |
| Plan limits | `services/quota_service.py` — checked before uploads, graph builds |
| Billing page | Plan cards, upgrade flow, current usage meters |
| Usage aggregation | Celery Beat job: daily usage stats |
| Email on quota exceeded | Send transactional email via SendGrid/Resend |

**Sprint 7 done when:** Free tier user hits doc limit → sees upgrade prompt → upgrades → limit lifted.

---

### SPRINT 8 — Production Hardening (Week 8)
**Goal: Deploy to production, observable, secure**

| Task | Files to Create |
|---|---|
| Dockerfile for each service | Multi-stage builds, minimal images |
| GitHub Actions CI | Test + lint on PR, build image on merge to main |
| Sentry integration | Error tracking in both API and Web |
| Structured logging | JSON logs with correlation IDs (api + worker) |
| Health check endpoints | `GET /health` (db + redis connectivity) |
| Rate limiting | Redis token bucket middleware |
| Terraform (optional) | Or deploy on Railway/Render for speed |
| Staging deploy | Full deploy, smoke tests passing |

**Sprint 8 done when:** App running on `app.aris.ai`, monitored, errors tracked, deploys automated.

---

## 9. ENVIRONMENT VARIABLES

```bash
# ── Database ──────────────────────────────────────────────
DATABASE_URL=postgresql+asyncpg://user:pass@localhost:5432/aris
DATABASE_POOL_SIZE=20
DATABASE_MAX_OVERFLOW=10

# ── Redis / Celery ────────────────────────────────────────
REDIS_URL=redis://localhost:6379/0
CELERY_BROKER_URL=redis://localhost:6379/0
CELERY_RESULT_BACKEND=redis://localhost:6379/1

# ── Auth ──────────────────────────────────────────────────
JWT_PRIVATE_KEY=<RS256 private key PEM — generate with openssl>
JWT_PUBLIC_KEY=<RS256 public key PEM>
JWT_ACCESS_TOKEN_EXPIRE_MINUTES=15
JWT_REFRESH_TOKEN_EXPIRE_DAYS=30

# ── AWS ───────────────────────────────────────────────────
AWS_ACCESS_KEY_ID=<from Secrets Manager in prod>
AWS_SECRET_ACCESS_KEY=<from Secrets Manager in prod>
AWS_REGION=ap-south-1
S3_BUCKET_NAME=aris-documents-prod
S3_PRESIGNED_UPLOAD_EXPIRY=600     # 10 minutes
S3_PRESIGNED_DOWNLOAD_EXPIRY=300   # 5 minutes

# ── Stripe ────────────────────────────────────────────────
STRIPE_SECRET_KEY=sk_live_...
STRIPE_WEBHOOK_SECRET=whsec_...
STRIPE_PRO_PRICE_ID=price_...
STRIPE_TEAM_PRICE_ID=price_...

# ── App ───────────────────────────────────────────────────
APP_ENV=production               # development | staging | production
APP_BASE_URL=https://api.aris.ai
FRONTEND_URL=https://app.aris.ai
CORS_ORIGINS=["https://app.aris.ai"]
LOG_LEVEL=INFO

# ── Email ─────────────────────────────────────────────────
RESEND_API_KEY=re_...
EMAIL_FROM=noreply@aris.ai

# ── Monitoring ────────────────────────────────────────────
SENTRY_DSN=https://...@sentry.io/...

# ── ARIS Core ─────────────────────────────────────────────
ARIS_MAX_CORPUS_DOCS=500         # Hard limit per graph build
ARIS_CONFIDENCE_THRESHOLD=0.5    # Default edge materialisation threshold
ARIS_MAX_ACTIONS=50              # Default max research actions
```

---

## 10. CODING AGENT INSTRUCTIONS

Copy this entire section verbatim into your agent's system prompt or task file.

### Agent Context
You are building ARIS Platform — a SaaS product wrapping the `aris-core` Python library. The full file structure and specifications are in this document. Follow them exactly.

### Absolute Rules
1. **Never modify** any file inside `packages/aris-core/aris/core/` or `packages/aris-core/aris/graph/`. These are sealed. If you need to extend ARIS behaviour, do it in `apps/api/services/`.
2. **Always use async** SQLAlchemy (AsyncSession, async with). Never use sync SQLAlchemy.
3. **Never put secrets** in code. Always read from `config.py` which loads from env.
4. **Always validate workspace ownership** before any data operation. Use `get_workspace_or_403()` from `dependencies.py`.
5. **Every API endpoint** must be covered by at least one test in `apps/api/tests/`.
6. **Never expose S3 keys to the frontend.** Always use pre-signed URLs.
7. **Never run ARIS core in an API request handler.** Always enqueue a Celery task.

### Build Order
Build in sprint order (Section 8). Do not skip sprints. Each sprint has a "done when" condition — verify it before starting the next.

### File Creation Protocol
For every new file: (1) create the file with full implementation, (2) add corresponding test, (3) update `__init__.py` exports if needed, (4) run `mypy` on backend files and `tsc --noEmit` on frontend files before marking complete.

### ARIS Core Usage Pattern
```python
# CORRECT — how to call aris-core from a service
from aris.graph.document_ingestion import DocumentIngestor, DocumentCorpus
from aris.graph.knowledge_graph import build_graph_from_corpus, Linker, LinkMaterializer
from aris.graph.research_planner import ResearchPlanner, PlannerContext
from aris.core.memory_store import FileBackedMemoryStore
from aris.core.run_loop import run_loop

# In graph_build_task.py:
async def build_graph(graph_id: UUID, document_s3_keys: list[str]):
    # 1. Download docs from S3, write to /tmp/
    # 2. Ingest with DocumentIngestor
    # 3. run build_graph_from_corpus → nodes
    # 4. Linker.generate_candidates → candidates
    # 5. LinkMaterializer.materialise → edges
    # 6. Persist nodes + edges to PostgreSQL
    # 7. Serialise traces to aris_traces table
    # 8. Update knowledge_graphs.status = 'completed'
```

---

## 11. QUICK-START COMMANDS FOR THE AGENT

```bash
# Bootstrap the repo
git clone https://github.com/JeyadeepakUR/ARIS-prod aris-platform
cd aris-platform
git subtree split --prefix=. -b core-only  # restructure for monorepo

# Or: start fresh monorepo and add core as subtree
mkdir aris-platform && cd aris-platform
git init
git subtree add --prefix=packages/aris-core https://github.com/JeyadeepakUR/ARIS-prod.git main --squash

# Start local dev stack
cp .env.example .env           # fill in values
make dev                       # docker-compose up --build

# Run API tests
cd apps/api && pytest -x -q

# Run frontend dev server
cd apps/web && npm install && npm run dev

# Generate a DB migration
cd apps/api && alembic revision --autogenerate -m "initial schema"
alembic upgrade head

# Type check everything
cd apps/api && mypy . --strict
cd apps/web && npx tsc --noEmit
```

---

## 12. KNOWN RISKS & MITIGATIONS

| Risk | Likelihood | Impact | Mitigation |
|---|---|---|---|
| ARIS pipeline too slow for UX (>2min) | Medium | High | Cap corpus size per build; show progress via SSE; optimise Linker for large N |
| aris-core sealed — can't fix bugs fast | Low | High | Fixes allowed per contract (bug-only); governance review takes <1 day solo |
| S3 costs spike on large uploads | Medium | Medium | File size limits per plan; S3 lifecycle to Glacier after 90 days |
| Celery worker OOM on 500-doc corpus | Medium | High | K8s memory limits; chunk large corpora into 50-doc batches in graph_build_task |
| JWT secret compromised | Low | Critical | RS256 asymmetric keys; private key in Secrets Manager only; rotate quarterly |
| Stripe webhook replay attacks | Low | High | Verify Stripe-Signature header; idempotency key on subscription events |
| Phase 2 ML breaks Phase 1 determinism | Medium | Critical | Enforced by Phase 1 contract + CI test suite; ML modules only in aris.ml |

---

*This document is the single source of truth for the ARIS Platform build. Every architectural decision is made here — the coding agent executes it.*