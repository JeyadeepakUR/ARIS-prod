"""
pgvector-backed semantic search.

All vector operations go through raw SQL (SQLAlchemy text()) using the
pgvector <=> cosine-distance operator. This avoids asyncpg type-registration
complexity while still using the ivfflat index on the vector column.

Vector literal format PostgreSQL expects:  '[0.1,0.2,...,0.768]'

On non-PostgreSQL backends (SQLite in tests), all pgvector-specific methods
return empty results gracefully rather than raising.
"""
from __future__ import annotations

import json
import logging
import uuid
from dataclasses import dataclass

from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession

logger = logging.getLogger(__name__)


@dataclass
class ChunkSearchResult:
    chunk_id: str
    document_id: str
    content: str
    score: float          # cosine similarity 0.0 – 1.0 (higher = more similar)
    page_number: int | None
    section: str | None
    metadata: dict


@dataclass
class NodeSearchResult:
    node_id: str
    label: str
    domain: str
    graph_id: str
    score: float


def _vec_literal(embedding: list[float]) -> str:
    """Convert a Python float list to a PostgreSQL vector literal string."""
    return "[" + ",".join(f"{x:.8f}" for x in embedding) + "]"


class VectorStore:
    """Insert and query embeddings in PostgreSQL via pgvector."""

    def __init__(self, session: AsyncSession) -> None:
        self._session = session
        self._pgvector: bool | None = None  # lazily determined

    async def _supports_pgvector(self) -> bool:
        """Return True if the current DB is PostgreSQL with pgvector available.

        Inspecting the engine dialect is safe and side-effect free; running a
        probe query that fails would force an aborted-transaction rollback
        on PostgreSQL, AND on SQLite a rollback would also discard any rows
        the caller just flushed in the same transaction (e.g. concept nodes).
        """
        if self._pgvector is not None:
            return self._pgvector
        try:
            bind = self._session.get_bind()
            dialect_name = getattr(bind.dialect, "name", "")
        except Exception:
            dialect_name = ""
        if dialect_name != "postgresql":
            self._pgvector = False
            return False
        # We're on PostgreSQL — probe pgvector inside a savepoint so a
        # failure doesn't leak into the outer transaction.
        try:
            async with self._session.begin_nested():
                await self._session.execute(text("SELECT 1::vector(1)"))
            self._pgvector = True
        except Exception:
            self._pgvector = False
        return self._pgvector

    # ── insert ───────────────────────────────────────────────────────────────

    async def insert_chunk(
        self,
        *,
        document_id: str,
        chunk_index: int,
        content: str,
        embedding: list[float],
        page_number: int | None = None,
        section: str | None = None,
        metadata: dict | None = None,
    ) -> str:
        """
        Insert (or upsert) a chunk with its embedding.
        Returns the chunk UUID as a string.
        Falls back to ORM insert without embedding on non-PostgreSQL backends.
        """
        chunk_id = str(uuid.uuid4())

        if not await self._supports_pgvector():
            # SQLite / test env — store text only via ORM
            from uuid import UUID

            from apps.api.models.chunk import DocumentChunk
            session = self._session
            existing = (await session.execute(
                text("SELECT id FROM document_chunks WHERE document_id=:d AND chunk_index=:i"),
                {"d": str(document_id), "i": chunk_index},
            )).fetchone()
            if existing:
                await session.execute(
                    text("UPDATE document_chunks SET content=:c, section=:s WHERE id=:id"),
                    {"c": content, "s": section, "id": str(existing[0])},
                )
                return str(existing[0])
            session.add(DocumentChunk(
                id=UUID(chunk_id),
                document_id=UUID(str(document_id)),
                chunk_index=chunk_index,
                content=content,
                page_number=page_number,
                section=section,
                metadata_json=metadata or {},
            ))
            return chunk_id

        vec = _vec_literal(embedding)
        meta_json = json.dumps(metadata or {})

        await self._session.execute(
            text("""
                INSERT INTO document_chunks
                    (id, document_id, chunk_index, content, embedding,
                     page_number, section, metadata)
                VALUES
                    (:id::uuid, :doc_id::uuid, :chunk_idx, :content, :vec::vector,
                     :page_number, :section, :meta::jsonb)
                ON CONFLICT (document_id, chunk_index) DO UPDATE
                    SET content   = EXCLUDED.content,
                        embedding = EXCLUDED.embedding,
                        section   = EXCLUDED.section,
                        metadata  = EXCLUDED.metadata
            """),
            {
                "id": chunk_id,
                "doc_id": str(document_id),
                "chunk_idx": chunk_index,
                "content": content,
                "vec": vec,
                "page_number": page_number,
                "section": section,
                "meta": meta_json,
            },
        )
        return chunk_id

    async def insert_chunk_batch(
        self,
        chunks: list[dict],
    ) -> list[str]:
        """
        Insert multiple chunks efficiently.
        Each dict must have: document_id, chunk_index, content, embedding.
        Optional keys: page_number, section, metadata.
        Returns list of chunk UUIDs.
        """
        ids: list[str] = []
        for c in chunks:
            chunk_id = await self.insert_chunk(
                document_id=c["document_id"],
                chunk_index=c["chunk_index"],
                content=c["content"],
                embedding=c["embedding"],
                page_number=c.get("page_number"),
                section=c.get("section"),
                metadata=c.get("metadata"),
            )
            ids.append(chunk_id)
        return ids

    # ── search ────────────────────────────────────────────────────────────────

    async def similarity_search(
        self,
        query_embedding: list[float],
        *,
        top_k: int = 10,
        document_ids: list[str] | None = None,
        min_score: float = 0.5,
    ) -> list[ChunkSearchResult]:
        """
        Return top_k chunks closest to query_embedding by cosine similarity.
        Returns empty list on non-PostgreSQL backends.
        """
        if not await self._supports_pgvector():
            return []

        vec = _vec_literal(query_embedding)

        if document_ids:
            doc_filter = "AND document_id::text = ANY(:doc_ids)"
            params: dict = {"vec": vec, "min_score": min_score, "top_k": top_k, "doc_ids": [str(d) for d in document_ids]}
        else:
            doc_filter = ""
            params = {"vec": vec, "min_score": min_score, "top_k": top_k}

        rows = await self._session.execute(
            text(f"""
                SELECT
                    id::text                                         AS chunk_id,
                    document_id::text                               AS document_id,
                    content,
                    1 - (embedding <=> :vec::vector)                AS score,
                    page_number,
                    section,
                    metadata
                FROM document_chunks
                WHERE embedding IS NOT NULL
                  {doc_filter}
                  AND 1 - (embedding <=> :vec::vector) >= :min_score
                ORDER BY embedding <=> :vec::vector
                LIMIT :top_k
            """),
            params,
        )

        return [
            ChunkSearchResult(
                chunk_id=row.chunk_id,
                document_id=row.document_id,
                content=row.content,
                score=float(row.score),
                page_number=row.page_number,
                section=row.section,
                metadata=row.metadata or {},
            )
            for row in rows
        ]

    async def node_similarity_search(
        self,
        query_embedding: list[float],
        *,
        top_k: int = 20,
        exclude_domain: str | None = None,
        graph_id: str | None = None,
        min_score: float = 0.72,
    ) -> list[NodeSearchResult]:
        """
        Find nodes semantically similar to query_embedding.
        Returns empty list on non-PostgreSQL backends.
        """
        if not await self._supports_pgvector():
            return []

        vec = _vec_literal(query_embedding)

        conditions = ["n.embedding IS NOT NULL"]
        params: dict = {"vec": vec, "min_score": min_score, "top_k": top_k}

        if exclude_domain:
            conditions.append("n.cluster_id != :exclude_domain")
            params["exclude_domain"] = exclude_domain

        if graph_id:
            conditions.append("n.graph_id = :graph_id::uuid")
            params["graph_id"] = graph_id

        where = " AND ".join(conditions)

        rows = await self._session.execute(
            text(f"""
                SELECT
                    n.id::text                                  AS node_id,
                    n.label,
                    COALESCE(n.cluster_id, 'unknown')          AS domain,
                    n.graph_id::text,
                    1 - (n.embedding <=> :vec::vector)         AS score
                FROM nodes n
                WHERE {where}
                  AND 1 - (n.embedding <=> :vec::vector) >= :min_score
                ORDER BY n.embedding <=> :vec::vector
                LIMIT :top_k
            """),
            params,
        )

        return [
            NodeSearchResult(
                node_id=row.node_id,
                label=row.label,
                domain=row.domain,
                graph_id=row.graph_id,
                score=float(row.score),
            )
            for row in rows
        ]

    async def update_node_embedding(self, node_id: str, embedding: list[float]) -> None:
        """Write (or overwrite) an embedding on a nodes row via raw SQL.
        No-op on non-PostgreSQL backends."""
        if not await self._supports_pgvector():
            return
        vec = _vec_literal(embedding)
        await self._session.execute(
            text("UPDATE nodes SET embedding = :vec::vector WHERE id = :node_id::uuid"),
            {"vec": vec, "node_id": node_id},
        )

    async def cross_domain_bridge_search(
        self,
        *,
        graph_id: str,
        min_score: float = 0.72,
        limit: int = 60,
    ) -> list[dict]:  # noqa: E501
        """
        Find all cross-domain node pairs within a graph that have cosine
        similarity >= min_score.  Returns dicts with node1/node2 metadata.
        Falls back to Python-side cosine on non-PostgreSQL backends using
        the inline embedding stored in nodes.metadata.embedding_inline.
        """
        if not await self._supports_pgvector():
            return await self._sqlite_pair_search(
                graph_id=graph_id,
                min_score=min_score,
                limit=limit,
                same_domain=False,
            )
        # Force a sequential scan: IVFFlat index is ANN-only (ORDER BY … LIMIT)
        # and doesn't help a self-join with a similarity filter — using it can
        # actually suppress results on small tables with many lists.
        await self._session.execute(text("SET LOCAL enable_indexscan = off"))
        rows = await self._session.execute(
            text("""
                SELECT
                    n1.id::text          AS node1_id,
                    n1.label             AS node1_label,
                    COALESCE(n1.cluster_id, 'unknown') AS domain1,
                    n2.id::text          AS node2_id,
                    n2.label             AS node2_label,
                    COALESCE(n2.cluster_id, 'unknown') AS domain2,
                    1 - (n1.embedding <=> n2.embedding) AS similarity
                FROM nodes n1
                JOIN nodes n2
                    ON  n1.graph_id = n2.graph_id
                    AND n1.id < n2.id
                    AND COALESCE(n1.cluster_id, 'x') != COALESCE(n2.cluster_id, 'x')
                WHERE n1.graph_id = :gid::uuid
                  AND n1.embedding IS NOT NULL
                  AND n2.embedding IS NOT NULL
                  AND 1 - (n1.embedding <=> n2.embedding) >= :min_score
                ORDER BY similarity DESC
                LIMIT :lim
            """),
            {"gid": graph_id, "min_score": min_score, "lim": limit},
        )
        await self._session.execute(text("SET LOCAL enable_indexscan = on"))
        return [dict(row._mapping) for row in rows]

    async def intra_domain_similarity_search(
        self,
        *,
        graph_id: str,
        min_score: float = 0.75,
        limit: int = 80,
    ) -> list[dict]:
        """
        Find high-similarity node pairs within the SAME domain for a graph.
        Used to build intra-domain edges when no cross-domain bridges exist.
        Falls back to Python-side cosine on non-PostgreSQL backends using
        nodes.metadata.embedding_inline.
        """
        if not await self._supports_pgvector():
            return await self._sqlite_pair_search(
                graph_id=graph_id,
                min_score=min_score,
                limit=limit,
                same_domain=True,
            )
        await self._session.execute(text("SET LOCAL enable_indexscan = off"))
        rows = await self._session.execute(
            text("""
                SELECT
                    n1.id::text          AS node1_id,
                    n1.label             AS node1_label,
                    COALESCE(n1.cluster_id, 'unknown') AS domain1,
                    n2.id::text          AS node2_id,
                    n2.label             AS node2_label,
                    COALESCE(n2.cluster_id, 'unknown') AS domain2,
                    1 - (n1.embedding <=> n2.embedding) AS similarity
                FROM nodes n1
                JOIN nodes n2
                    ON  n1.graph_id = n2.graph_id
                    AND n1.id < n2.id
                    AND COALESCE(n1.cluster_id, 'x') = COALESCE(n2.cluster_id, 'x')
                WHERE n1.graph_id = :gid::uuid
                  AND n1.embedding IS NOT NULL
                  AND n2.embedding IS NOT NULL
                  AND 1 - (n1.embedding <=> n2.embedding) >= :min_score
                ORDER BY similarity DESC
                LIMIT :lim
            """),
            {"gid": graph_id, "min_score": min_score, "lim": limit},
        )
        await self._session.execute(text("SET LOCAL enable_indexscan = on"))
        return [dict(row._mapping) for row in rows]

    async def cross_document_similar_chunks(
        self,
        *,
        document_ids: list[str],
        min_score: float = 0.80,
        limit: int = 40,
    ) -> list[dict]:
        """
        Find semantically similar chunk pairs from *different* documents.
        Used by contradiction_analyst to surface candidate contradictions.
        Returns empty list on non-PostgreSQL backends without inline embeddings.
        """
        if not await self._supports_pgvector():
            # SQLite path: chunks don't carry inline embeddings (would bloat
            # the DB). We surface a coarse fallback: pair the first chunk of
            # each document with the first chunk of every other document so
            # the contradiction analyst still has SOMETHING to evaluate
            # locally. Quality is much better with PostgreSQL+pgvector.
            return await self._sqlite_chunk_pairs(
                document_ids=document_ids,
                limit=limit,
            )
        rows = await self._session.execute(
            text("""
                SELECT
                    c1.id::text              AS chunk1_id,
                    c1.document_id::text     AS doc1_id,
                    c1.content               AS content1,
                    c2.id::text              AS chunk2_id,
                    c2.document_id::text     AS doc2_id,
                    c2.content               AS content2,
                    1 - (c1.embedding <=> c2.embedding) AS similarity
                FROM document_chunks c1
                JOIN document_chunks c2
                    ON  c1.document_id != c2.document_id
                    AND c1.id < c2.id
                WHERE c1.document_id::text = ANY(:doc_ids)
                  AND c2.document_id::text = ANY(:doc_ids)
                  AND c1.embedding IS NOT NULL
                  AND c2.embedding IS NOT NULL
                  AND 1 - (c1.embedding <=> c2.embedding) >= :min_score
                ORDER BY similarity DESC
                LIMIT :lim
            """),
            {"doc_ids": [str(d) for d in document_ids], "min_score": min_score, "lim": limit},
        )
        return [dict(row._mapping) for row in rows]

    async def find_graph_gaps(
        self,
        *,
        graph_id: str,
        min_shared: int = 2,
        limit: int = 20,
    ) -> list[dict]:
        """
        Node pairs with >= min_shared common neighbors but no direct edge.
        These represent structural research gaps in the knowledge graph.
        Falls back to Python-side computation on non-PostgreSQL backends.
        """
        if not await self._supports_pgvector():
            return await self._sqlite_find_gaps(
                graph_id=graph_id, min_shared=min_shared, limit=limit
            )
        rows = await self._session.execute(
            text("""
                SELECT
                    e1.source_node_id::text  AS n1_id,
                    e2.source_node_id::text  AS n2_id,
                    n1.label                 AS n1_label,
                    n2.label                 AS n2_label,
                    COUNT(DISTINCT e1.target_node_id) AS shared_count
                FROM edges e1
                JOIN edges e2
                    ON  e1.target_node_id = e2.target_node_id
                    AND e1.source_node_id < e2.source_node_id
                JOIN nodes n1 ON n1.id = e1.source_node_id
                JOIN nodes n2 ON n2.id = e2.source_node_id
                WHERE e1.graph_id = :gid::uuid
                  AND e2.graph_id = :gid::uuid
                  AND NOT EXISTS (
                      SELECT 1 FROM edges d
                      WHERE d.graph_id = :gid::uuid
                        AND (
                            (d.source_node_id = e1.source_node_id AND d.target_node_id = e2.source_node_id)
                         OR (d.source_node_id = e2.source_node_id AND d.target_node_id = e1.source_node_id)
                        )
                  )
                GROUP BY e1.source_node_id, e2.source_node_id, n1.label, n2.label
                HAVING COUNT(DISTINCT e1.target_node_id) >= :min_shared
                ORDER BY shared_count DESC
                LIMIT :lim
            """),
            {"gid": graph_id, "min_shared": min_shared, "lim": limit},
        )
        return [dict(row._mapping) for row in rows]

    async def delete_chunks_for_document(self, document_id: str) -> int:
        """Delete all chunks for a document. Returns deleted count."""
        if await self._supports_pgvector():
            result = await self._session.execute(
                text("DELETE FROM document_chunks WHERE document_id = :doc_id::uuid"),
                {"doc_id": str(document_id)},
            )
        else:
            result = await self._session.execute(
                text("DELETE FROM document_chunks WHERE document_id = :doc_id"),
                {"doc_id": str(document_id)},
            )
        return result.rowcount  # type: ignore[union-attr]

    # ── SQLite (and other non-pgvector) fallbacks ────────────────────────────

    async def _sqlite_pair_search(
        self,
        *,
        graph_id: str,
        min_score: float,
        limit: int,
        same_domain: bool,
    ) -> list[dict]:
        """Python-side cosine similarity over nodes.metadata.embedding_inline.

        Concept-extractor stores each concept's embedding inline in metadata
        precisely so this fallback can produce edges without pgvector.
        """
        from sqlalchemy import select as sa_select

        from apps.api.models.node import Node as NodeModel
        rows = await self._session.execute(
            sa_select(
                NodeModel.id,
                NodeModel.label,
                NodeModel.cluster_id,
                NodeModel.metadata_json,
                NodeModel.node_type,
            ).where(
                NodeModel.graph_id == _to_uuid(graph_id),
                NodeModel.node_type == "concept",
            )
        )
        candidates: list[tuple[str, str, str, list[float]]] = []
        for r in rows.fetchall():
            meta = r.metadata_json or {}
            emb = meta.get("embedding_inline")
            if not emb or not isinstance(emb, list):
                continue
            candidates.append((str(r.id), r.label, r.cluster_id or "unknown", emb))

        results: list[dict] = []
        for i in range(len(candidates)):
            id_i, label_i, dom_i, emb_i = candidates[i]
            for j in range(i + 1, len(candidates)):
                id_j, label_j, dom_j, emb_j = candidates[j]
                if same_domain and dom_i != dom_j:
                    continue
                if not same_domain and dom_i == dom_j:
                    continue
                sim = _cosine_py(emb_i, emb_j)
                if sim < min_score:
                    continue
                results.append({
                    "node1_id": id_i,
                    "node1_label": label_i,
                    "domain1": dom_i,
                    "node2_id": id_j,
                    "node2_label": label_j,
                    "domain2": dom_j,
                    "similarity": sim,
                })
        results.sort(key=lambda r: r["similarity"], reverse=True)
        return results[:limit]

    async def _sqlite_chunk_pairs(
        self,
        *,
        document_ids: list[str],
        limit: int,
    ) -> list[dict]:
        """Coarse cross-document chunk pairing for the SQLite/no-pgvector path.

        Pairs the first chunk of each document with the first chunk of each
        other document so the contradiction analyst can still operate locally.
        """
        from sqlalchemy import select as sa_select

        from apps.api.models.chunk import DocumentChunk
        first_chunks: list[tuple[str, str, str]] = []
        for d in document_ids:
            row = (await self._session.execute(
                sa_select(DocumentChunk.id, DocumentChunk.document_id, DocumentChunk.content)
                .where(DocumentChunk.document_id == _to_uuid(d))
                .order_by(DocumentChunk.chunk_index)
                .limit(1)
            )).first()
            if row is None:
                continue
            first_chunks.append((str(row[0]), str(row[1]), row[2] or ""))

        out: list[dict] = []
        for i in range(len(first_chunks)):
            for j in range(i + 1, len(first_chunks)):
                a = first_chunks[i]
                b = first_chunks[j]
                if a[1] == b[1]:
                    continue
                out.append({
                    "chunk1_id": a[0],
                    "doc1_id": a[1],
                    "content1": a[2],
                    "chunk2_id": b[0],
                    "doc2_id": b[1],
                    "content2": b[2],
                    "similarity": 0.5,
                })
        return out[:limit]

    async def _sqlite_find_gaps(
        self,
        *,
        graph_id: str,
        min_shared: int,
        limit: int,
    ) -> list[dict]:
        """Pure-Python structural-gap detection for non-pgvector backends."""
        from sqlalchemy import select as sa_select

        from apps.api.models.edge import Edge as EdgeModel
        from apps.api.models.node import Node as NodeModel

        edge_rows = await self._session.execute(
            sa_select(EdgeModel.source_node_id, EdgeModel.target_node_id)
            .where(EdgeModel.graph_id == _to_uuid(graph_id))
        )
        adj: dict[str, set[str]] = {}
        all_pairs: set[frozenset[str]] = set()
        for src, tgt in edge_rows.fetchall():
            s = str(src)
            t = str(tgt)
            adj.setdefault(s, set()).add(t)
            adj.setdefault(t, set()).add(s)
            all_pairs.add(frozenset({s, t}))

        node_rows = await self._session.execute(
            sa_select(NodeModel.id, NodeModel.label)
            .where(NodeModel.graph_id == _to_uuid(graph_id))
        )
        labels = {str(r.id): r.label for r in node_rows.fetchall()}

        candidates: list[dict] = []
        node_ids = sorted(adj.keys())
        for i in range(len(node_ids)):
            for j in range(i + 1, len(node_ids)):
                a = node_ids[i]
                b = node_ids[j]
                if frozenset({a, b}) in all_pairs:
                    continue
                shared = adj[a] & adj[b]
                if len(shared) >= min_shared:
                    candidates.append({
                        "n1_id": a,
                        "n2_id": b,
                        "n1_label": labels.get(a, a[:8]),
                        "n2_label": labels.get(b, b[:8]),
                        "shared_count": len(shared),
                    })
        candidates.sort(key=lambda r: r["shared_count"], reverse=True)
        return candidates[:limit]


def _cosine_py(a: list[float], b: list[float]) -> float:
    if len(a) != len(b) or not a:
        return 0.0
    dot = 0.0
    na = 0.0
    nb = 0.0
    for x, y in zip(a, b):
        dot += x * y
        na += x * x
        nb += y * y
    if na == 0.0 or nb == 0.0:
        return 0.0
    return dot / ((na ** 0.5) * (nb ** 0.5))


def _to_uuid(val: str) -> object:
    """Coerce string UUIDs to uuid.UUID for SQLAlchemy ORM filters."""
    try:
        return uuid.UUID(str(val))
    except (TypeError, ValueError):
        return val
