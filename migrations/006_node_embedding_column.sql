-- Migration 006: ensure nodes.embedding column exists.
-- 003_pgvector.sql ran before ORM create_all() created the nodes table,
-- so the ALTER TABLE in that migration was a no-op.

ALTER TABLE nodes ADD COLUMN IF NOT EXISTS embedding vector(768);

DO $$
BEGIN
    IF NOT EXISTS (SELECT 1 FROM pg_indexes WHERE indexname = 'nodes_embedding_idx') THEN
        CREATE INDEX nodes_embedding_idx
            ON nodes USING ivfflat (embedding vector_cosine_ops) WITH (lists = 50);
    END IF;
END $$;
