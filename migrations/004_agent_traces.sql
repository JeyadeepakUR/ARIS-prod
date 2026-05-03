-- Migration 004: Agent execution traces + contradiction table
-- Handles both ORM table names (graphs/nodes/edges) and legacy SQL names
-- (knowledge_graphs/graph_nodes/graph_edges) so this works on both
-- Docker-compose (001_initial.sql schema) and Alembic-managed installs.

-- Per-step trace of what each LangGraph agent node did
-- thread_id maps to a LangGraph checkpoint thread (one per graph build run)
-- FK to graphs table added via DO block below to handle name variants.
CREATE TABLE IF NOT EXISTS agent_traces (
    id           UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    graph_id     UUID NOT NULL,
    thread_id    TEXT NOT NULL,
    agent_node   TEXT NOT NULL,
    step_index   INTEGER NOT NULL,
    input_state  JSONB NOT NULL DEFAULT '{}',
    output_state JSONB NOT NULL DEFAULT '{}',
    reasoning    TEXT,
    tokens_used  INTEGER,
    duration_ms  INTEGER,
    created_at   TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS agent_traces_graph_id_idx ON agent_traces (graph_id);
CREATE INDEX IF NOT EXISTS agent_stream_events_thread_id_idx ON agent_traces (thread_id);

-- Explicit contradiction table
-- Contradictions are between two claims (edges) from different authors/documents
-- FK columns are nullable; constraints added via DO block below.
CREATE TABLE IF NOT EXISTS contradictions (
    id                  UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    graph_id            UUID NOT NULL,
    claim_a_edge_id     UUID,
    claim_b_edge_id     UUID,
    claim_a_chunk_id    UUID REFERENCES document_chunks(id) ON DELETE SET NULL,
    claim_b_chunk_id    UUID REFERENCES document_chunks(id) ON DELETE SET NULL,
    author_a            TEXT,
    author_b            TEXT,
    claim_a_text        TEXT NOT NULL,
    claim_b_text        TEXT NOT NULL,
    contradiction_type  TEXT NOT NULL DEFAULT 'direct',
    -- direct: same topic, opposite conclusions
    -- methodological: same topic, different methods, different outcomes
    -- scope: one claim is a subset/superset that conflicts with the other
    severity            FLOAT NOT NULL DEFAULT 0.5 CHECK (severity BETWEEN 0 AND 1),
    llm_reasoning       TEXT,
    status              TEXT NOT NULL DEFAULT 'open'
                            CHECK (status IN ('open', 'resolved', 'disputed')),
    created_at          TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS contradictions_graph_id_idx ON contradictions (graph_id);

-- Stream events emitted by agents during a run (consumed by SSE endpoint)
-- Rows are short-lived; cleaned up after graph build completes
CREATE TABLE IF NOT EXISTS agent_stream_events (
    id          UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    graph_id    UUID NOT NULL,
    thread_id   TEXT NOT NULL,
    agent_node  TEXT NOT NULL,
    event_type  TEXT NOT NULL,
    -- concept_found | bridge_found | contradiction_found | hypothesis_generated
    -- gap_identified | step_complete | run_complete | error
    content     JSONB NOT NULL DEFAULT '{}',
    created_at  TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS agent_stream_events_graph_id_idx
    ON agent_stream_events (graph_id, created_at);

-- Add FK constraints pointing at whichever graphs table variant exists
DO $$
DECLARE
    graphs_table TEXT;
    edges_table  TEXT;
BEGIN
    IF EXISTS (SELECT 1 FROM information_schema.tables WHERE table_name = 'graphs') THEN
        graphs_table := 'graphs';
    ELSIF EXISTS (SELECT 1 FROM information_schema.tables WHERE table_name = 'knowledge_graphs') THEN
        graphs_table := 'knowledge_graphs';
    END IF;

    IF EXISTS (SELECT 1 FROM information_schema.tables WHERE table_name = 'edges') THEN
        edges_table := 'edges';
    ELSIF EXISTS (SELECT 1 FROM information_schema.tables WHERE table_name = 'graph_edges') THEN
        edges_table := 'graph_edges';
    END IF;

    IF graphs_table IS NOT NULL THEN
        IF NOT EXISTS (
            SELECT 1 FROM information_schema.table_constraints
            WHERE constraint_name = 'agent_traces_graph_id_fk'
        ) THEN
            EXECUTE format(
                'ALTER TABLE agent_traces ADD CONSTRAINT agent_traces_graph_id_fk '
                'FOREIGN KEY (graph_id) REFERENCES %I(id) ON DELETE CASCADE',
                graphs_table
            );
        END IF;

        IF NOT EXISTS (
            SELECT 1 FROM information_schema.table_constraints
            WHERE constraint_name = 'contradictions_graph_id_fk'
        ) THEN
            EXECUTE format(
                'ALTER TABLE contradictions ADD CONSTRAINT contradictions_graph_id_fk '
                'FOREIGN KEY (graph_id) REFERENCES %I(id) ON DELETE CASCADE',
                graphs_table
            );
        END IF;

        IF NOT EXISTS (
            SELECT 1 FROM information_schema.table_constraints
            WHERE constraint_name = 'agent_stream_events_graph_id_fk'
        ) THEN
            EXECUTE format(
                'ALTER TABLE agent_stream_events ADD CONSTRAINT agent_stream_events_graph_id_fk '
                'FOREIGN KEY (graph_id) REFERENCES %I(id) ON DELETE CASCADE',
                graphs_table
            );
        END IF;
    END IF;

    IF edges_table IS NOT NULL THEN
        IF NOT EXISTS (
            SELECT 1 FROM information_schema.table_constraints
            WHERE constraint_name = 'contradictions_claim_a_edge_fk'
        ) THEN
            EXECUTE format(
                'ALTER TABLE contradictions ADD CONSTRAINT contradictions_claim_a_edge_fk '
                'FOREIGN KEY (claim_a_edge_id) REFERENCES %I(id) ON DELETE SET NULL',
                edges_table
            );
        END IF;

        IF NOT EXISTS (
            SELECT 1 FROM information_schema.table_constraints
            WHERE constraint_name = 'contradictions_claim_b_edge_fk'
        ) THEN
            EXECUTE format(
                'ALTER TABLE contradictions ADD CONSTRAINT contradictions_claim_b_edge_fk '
                'FOREIGN KEY (claim_b_edge_id) REFERENCES %I(id) ON DELETE SET NULL',
                edges_table
            );
        END IF;
    END IF;
END $$;
