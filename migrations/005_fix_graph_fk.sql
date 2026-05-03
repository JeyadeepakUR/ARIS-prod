-- Migration 005: ensure the `graphs` table exists and all FK constraints point to it.
--
-- Root cause: 004_agent_traces.sql runs at postgres init time, before the API's
-- create_all() runs. At that point only `knowledge_graphs` exists (from 001_initial.sql),
-- so the DO-block in 004 added FKs pointing to `knowledge_graphs`. This migration
-- creates `graphs` first, then re-wires those constraints.

-- Create the canonical graphs table (idempotent — ORM's create_all also does this)
CREATE TABLE IF NOT EXISTS graphs (
    id           UUID PRIMARY KEY,
    workspace_id UUID NOT NULL REFERENCES workspaces(id) ON DELETE CASCADE,
    status       VARCHAR(50) NOT NULL DEFAULT 'pending',
    metadata     JSON NOT NULL DEFAULT '{}',
    created_at   TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS ix_graphs_workspace_id ON graphs (workspace_id);
CREATE INDEX IF NOT EXISTS ix_graphs_status       ON graphs (status);

-- Re-wire FK constraints that 004 may have pointed at knowledge_graphs
DO $$
BEGIN
    -- agent_stream_events
    ALTER TABLE agent_stream_events DROP CONSTRAINT IF EXISTS agent_stream_events_graph_id_fk;
    ALTER TABLE agent_stream_events ADD CONSTRAINT agent_stream_events_graph_id_fk
        FOREIGN KEY (graph_id) REFERENCES graphs(id) ON DELETE CASCADE;

    -- agent_traces
    ALTER TABLE agent_traces DROP CONSTRAINT IF EXISTS agent_traces_graph_id_fk;
    ALTER TABLE agent_traces ADD CONSTRAINT agent_traces_graph_id_fk
        FOREIGN KEY (graph_id) REFERENCES graphs(id) ON DELETE CASCADE;

    -- contradictions
    ALTER TABLE contradictions DROP CONSTRAINT IF EXISTS contradictions_graph_id_fk;
    ALTER TABLE contradictions ADD CONSTRAINT contradictions_graph_id_fk
        FOREIGN KEY (graph_id) REFERENCES graphs(id) ON DELETE CASCADE;
END $$;
