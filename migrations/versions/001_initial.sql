-- Initial ARIS platform schema

CREATE EXTENSION IF NOT EXISTS pgcrypto;

-- Users & Auth
CREATE TABLE users (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    email VARCHAR(255) UNIQUE NOT NULL,
    hashed_pw VARCHAR(255) NOT NULL,
    plan VARCHAR(50) NOT NULL DEFAULT 'free',
    stripe_customer_id VARCHAR(255),
    is_verified BOOLEAN DEFAULT FALSE,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE TABLE refresh_tokens (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id UUID NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    token_hash VARCHAR(255) UNIQUE NOT NULL,
    expires_at TIMESTAMPTZ NOT NULL,
    created_at TIMESTAMPTZ DEFAULT NOW()
);

-- Workspaces
CREATE TABLE workspaces (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    owner_id UUID NOT NULL REFERENCES users(id),
    name VARCHAR(255) NOT NULL,
    slug VARCHAR(255) UNIQUE NOT NULL,
    settings JSONB DEFAULT '{}',
    created_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE TABLE workspace_members (
    workspace_id UUID REFERENCES workspaces(id) ON DELETE CASCADE,
    user_id UUID REFERENCES users(id) ON DELETE CASCADE,
    role VARCHAR(50) NOT NULL DEFAULT 'viewer',
    PRIMARY KEY (workspace_id, user_id)
);

-- Documents
CREATE TABLE documents (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    workspace_id UUID NOT NULL REFERENCES workspaces(id) ON DELETE CASCADE,
    filename VARCHAR(500) NOT NULL,
    s3_key VARCHAR(1000) NOT NULL,
    file_format VARCHAR(50) NOT NULL,
    file_size_bytes BIGINT,
    status VARCHAR(50) NOT NULL DEFAULT 'pending',
    error_msg TEXT,
    metadata JSONB DEFAULT '{}',
    created_at TIMESTAMPTZ DEFAULT NOW()
);

-- Knowledge Graphs
CREATE TABLE knowledge_graphs (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    workspace_id UUID NOT NULL REFERENCES workspaces(id) ON DELETE CASCADE,
    name VARCHAR(255) NOT NULL,
    status VARCHAR(50) NOT NULL DEFAULT 'pending',
    node_count INT DEFAULT 0,
    edge_count INT DEFAULT 0,
    config JSONB DEFAULT '{}',
    created_at TIMESTAMPTZ DEFAULT NOW(),
    completed_at TIMESTAMPTZ
);

CREATE TABLE graph_nodes (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    graph_id UUID NOT NULL REFERENCES knowledge_graphs(id) ON DELETE CASCADE,
    document_id UUID REFERENCES documents(id),
    label VARCHAR(500) NOT NULL,
    node_type VARCHAR(100),
    metadata JSONB DEFAULT '{}'
);

CREATE TABLE graph_edges (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    graph_id UUID NOT NULL REFERENCES knowledge_graphs(id) ON DELETE CASCADE,
    source_node_id UUID NOT NULL REFERENCES graph_nodes(id),
    target_node_id UUID NOT NULL REFERENCES graph_nodes(id),
    edge_type VARCHAR(100) NOT NULL,
    confidence FLOAT NOT NULL CHECK (confidence >= 0 AND confidence <= 1),
    evidence JSONB NOT NULL,
    reasoning_trace_id UUID NOT NULL,
    metadata JSONB DEFAULT '{}'
);

-- ARIS Core Traces
CREATE TABLE aris_traces (
    id UUID PRIMARY KEY,
    graph_id UUID REFERENCES knowledge_graphs(id),
    trace_json JSONB NOT NULL,
    created_at TIMESTAMPTZ DEFAULT NOW()
);

-- Research Plans
CREATE TABLE research_actions (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    graph_id UUID NOT NULL REFERENCES knowledge_graphs(id) ON DELETE CASCADE,
    action_type VARCHAR(100) NOT NULL,
    priority FLOAT NOT NULL,
    rationale TEXT NOT NULL,
    evidence JSONB NOT NULL,
    status VARCHAR(50) DEFAULT 'proposed',
    created_at TIMESTAMPTZ DEFAULT NOW()
);

-- Async Jobs
CREATE TABLE async_jobs (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    workspace_id UUID REFERENCES workspaces(id),
    job_type VARCHAR(100) NOT NULL,
    status VARCHAR(50) NOT NULL DEFAULT 'queued',
    celery_task_id VARCHAR(255),
    payload JSONB DEFAULT '{}',
    result JSONB DEFAULT '{}',
    trace JSON NOT NULL DEFAULT '{}',
    error TEXT,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    started_at TIMESTAMPTZ,
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
