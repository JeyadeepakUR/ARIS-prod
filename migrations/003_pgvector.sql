-- Migration 003: pgvector extension + document_chunks + embedding columns
-- Handles both ORM table names (graphs/nodes/edges) and legacy SQL names
-- (knowledge_graphs/graph_nodes/graph_edges) so this works on both
-- Docker-compose (001_initial.sql schema) and Alembic-managed installs.

CREATE EXTENSION IF NOT EXISTS vector;

-- Stores chunked text from documents with semantic embeddings
CREATE TABLE IF NOT EXISTS document_chunks (
    id           UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    document_id  UUID NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
    chunk_index  INTEGER NOT NULL,
    content      TEXT NOT NULL,
    embedding    vector(768),
    page_number  INTEGER,
    section      TEXT,
    metadata     JSONB NOT NULL DEFAULT '{}',
    created_at   TIMESTAMPTZ NOT NULL DEFAULT now(),
    UNIQUE (document_id, chunk_index)
);

CREATE INDEX IF NOT EXISTS document_chunks_embedding_idx
    ON document_chunks
    USING ivfflat (embedding vector_cosine_ops)
    WITH (lists = 100);

-- Add embedding columns to nodes/edges — handles both table-name variants
DO $$
BEGIN
    -- nodes table (ORM-managed install)
    IF EXISTS (SELECT 1 FROM information_schema.tables WHERE table_name = 'nodes') THEN
        IF NOT EXISTS (
            SELECT 1 FROM information_schema.columns
            WHERE table_name = 'nodes' AND column_name = 'embedding'
        ) THEN
            ALTER TABLE nodes ADD COLUMN embedding vector(768);
        END IF;
    END IF;

    -- graph_nodes table (legacy Docker SQL install)
    IF EXISTS (SELECT 1 FROM information_schema.tables WHERE table_name = 'graph_nodes') THEN
        IF NOT EXISTS (
            SELECT 1 FROM information_schema.columns
            WHERE table_name = 'graph_nodes' AND column_name = 'embedding'
        ) THEN
            ALTER TABLE graph_nodes ADD COLUMN embedding vector(768);
        END IF;
    END IF;

    -- edges table (ORM-managed install)
    IF EXISTS (SELECT 1 FROM information_schema.tables WHERE table_name = 'edges') THEN
        IF NOT EXISTS (
            SELECT 1 FROM information_schema.columns
            WHERE table_name = 'edges' AND column_name = 'embedding'
        ) THEN
            ALTER TABLE edges ADD COLUMN embedding vector(768);
        END IF;
    END IF;

    -- graph_edges table (legacy Docker SQL install)
    IF EXISTS (SELECT 1 FROM information_schema.tables WHERE table_name = 'graph_edges') THEN
        IF NOT EXISTS (
            SELECT 1 FROM information_schema.columns
            WHERE table_name = 'graph_edges' AND column_name = 'embedding'
        ) THEN
            ALTER TABLE graph_edges ADD COLUMN embedding vector(768);
        END IF;
    END IF;
END $$;

-- IVFFlat index on whichever nodes table exists
DO $$
BEGIN
    IF EXISTS (SELECT 1 FROM information_schema.tables WHERE table_name = 'nodes') THEN
        IF NOT EXISTS (SELECT 1 FROM pg_indexes WHERE indexname = 'nodes_embedding_idx') THEN
            EXECUTE $q$
                CREATE INDEX nodes_embedding_idx
                    ON nodes USING ivfflat (embedding vector_cosine_ops) WITH (lists = 50)
            $q$;
        END IF;
    END IF;

    IF EXISTS (SELECT 1 FROM information_schema.tables WHERE table_name = 'graph_nodes') THEN
        IF NOT EXISTS (SELECT 1 FROM pg_indexes WHERE indexname = 'graph_nodes_embedding_idx') THEN
            EXECUTE $q$
                CREATE INDEX graph_nodes_embedding_idx
                    ON graph_nodes USING ivfflat (embedding vector_cosine_ops) WITH (lists = 50)
            $q$;
        END IF;
    END IF;
END $$;
