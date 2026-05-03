"""
Concept Extractor agent node.

Extracts named scientific concepts from document chunks via the LLM, then:
  1. normalises labels (canonical form, generic-term blocklist)
  2. cross-document deduplicates near-duplicate concepts via embedding similarity
  3. creates domain hub nodes + concept nodes + document nodes
  4. wires has_concept (domain → concept) and extracted_from (concept → document)
     edges so the frontend has hierarchical structure to render.

Dependencies injected through LangGraph config["configurable"]:
    session      — AsyncSession (SQLAlchemy)
    embedder     — aris.ingestion.embedder.Embedder
    llm_provider — aris.llm.provider.LLMProvider
"""
from __future__ import annotations

import asyncio
import json
import logging
import re
from datetime import UTC, datetime
from uuid import UUID, uuid4

from langchain_core.runnables import RunnableConfig

from apps.api.models.edge import Edge as EdgeModel
from apps.api.models.node import Node
from aris.agents.state import ConceptNode, ResearchState, StreamEvent
from aris.retrieval.vector_store import VectorStore

logger = logging.getLogger(__name__)


# ── Fixed domain vocabulary ─────────────────────────────────────────────────
# Locking the LLM to a closed set of domains is critical: free-form domain
# strings cause near-identical concepts to land in different cluster_ids,
# preventing cross-domain bridges from forming.
ALLOWED_DOMAINS: list[str] = [
    "Machine Learning",
    "Deep Learning",
    "Natural Language Processing",
    "Computer Vision",
    "Reinforcement Learning",
    "Cybersecurity",
    "Blockchain",
    "Healthcare",
    "Bioinformatics",
    "Robotics",
    "Information Retrieval",
    "Data Engineering",
    "Distributed Systems",
    "Software Engineering",
    "Statistics",
    "Optimization",
    "Quantum Computing",
    "General Research",
]

_GENERIC_BLOCKLIST: set[str] = {
    "data", "method", "model", "system", "approach", "result", "results",
    "performance", "technique", "application", "analysis", "framework",
    "study", "paper", "research", "experiment", "experiments", "algorithm",
    "process", "task", "feature", "features", "training", "testing",
    "evaluation", "metric", "metrics", "function", "functions", "value",
    "values", "input", "output", "dataset", "datasets", "table", "figure",
}


# ── Prompt with few-shot examples and strict schema ─────────────────────────
_EXTRACTION_PROMPT_TEMPLATE = """\
You are an expert scientific knowledge extractor. From the research excerpt(s) \
below, extract the most important NAMED scientific concepts: methods, models, \
algorithms, datasets, named techniques, theories, or named phenomena.

# Domain vocabulary (use EXACTLY one of these labels, never invent new ones)
{domains}

# Rules
- Extract 4-10 concepts per excerpt. Skip if there are fewer specific concepts.
- A concept is a SPECIFIC named entity (e.g. "Transformer", "BERT", "Federated \
Averaging", "Differential Privacy", "Convolutional Neural Network").
- NEVER extract generic words alone. Banned standalone labels: \
data, method, model, system, approach, result, performance, technique, \
analysis, framework, study, algorithm, process, task, feature, training, \
testing, evaluation.
- Label = canonical form, 3-80 characters, Title Case for multi-word names, \
preserve standard acronyms (e.g. "BERT", "GAN", "TLS").
- Strip leading articles ("the", "a", "an") and trailing punctuation.
- Choose the SINGLE best-matching domain from the vocabulary above.
- confidence is your certainty the concept is named, specific, and correctly \
assigned (0.0-1.0).

# Few-shot examples
Excerpt: "We fine-tune BERT on a medical corpus and compare against BioGPT \
using contrastive loss for clinical entity recognition."
JSON:
{{"concepts": [
  {{"label": "BERT", "domain": "Natural Language Processing", "confidence": 0.95}},
  {{"label": "BioGPT", "domain": "Healthcare", "confidence": 0.9}},
  {{"label": "Contrastive Loss", "domain": "Deep Learning", "confidence": 0.85}},
  {{"label": "Clinical Entity Recognition", "domain": "Healthcare", "confidence": 0.85}}
]}}

Excerpt: "Federated Averaging is combined with differential privacy noise \
injection on the CIFAR-10 benchmark."
JSON:
{{"concepts": [
  {{"label": "Federated Averaging", "domain": "Machine Learning", "confidence": 0.95}},
  {{"label": "Differential Privacy", "domain": "Cybersecurity", "confidence": 0.95}},
  {{"label": "CIFAR-10", "domain": "Computer Vision", "confidence": 0.9}}
]}}

# Excerpt(s) to extract from
{text}

Return ONLY a single JSON object of the form: \
{{"concepts": [{{"label": "...", "domain": "...", "confidence": 0.0-1.0}}]}}"""


# Tunables (tuned for OpenRouter free-tier rate limits ~20 req/min)
_CHUNK_CHARS_PER_BATCH = 2400   # large per-batch context to compensate for fewer calls
_CHUNKS_PER_BATCH = 4           # 4 × 2400 = ~9.6k chars / call
_MAX_CHUNKS_TOTAL = 60          # ~15 LLM calls per paper at 4 chunks/batch
_DEDUP_SIMILARITY = 0.88        # cosine threshold for cross-document dedup


def _now_iso() -> str:
    return datetime.now(UTC).isoformat()


def _stream_event(event_type: str, content: dict) -> StreamEvent:
    return {
        "agent_node": "concept_extractor",
        "event_type": event_type,
        "content": content,
        "timestamp": _now_iso(),
    }


def _normalize_label(raw: str) -> str:
    """Trim, strip articles & weird punctuation, collapse whitespace."""
    label = raw.strip().strip(".,;:!?'\"()[]{}")
    label = re.sub(r"^(the|a|an)\s+", "", label, flags=re.IGNORECASE)
    label = re.sub(r"\s+", " ", label)
    return label[:200]


def _is_low_value(label: str) -> bool:
    """Reject single common words and pure punctuation."""
    norm = label.lower().strip()
    if len(norm) < 3:
        return True
    if norm in _GENERIC_BLOCKLIST:
        return True
    # Single short word that's also a generic
    if " " not in norm and norm in _GENERIC_BLOCKLIST:
        return True
    if not re.search(r"[A-Za-z]", label):
        return True
    return False


def _normalize_domain(raw: str) -> str:
    """Map LLM-returned domain string to closest entry in ALLOWED_DOMAINS."""
    if not raw:
        return "General Research"
    candidate = raw.strip()
    # Exact match
    for d in ALLOWED_DOMAINS:
        if candidate.lower() == d.lower():
            return d
    # Partial match: look for any domain that contains or is contained by candidate
    cl = candidate.lower()
    for d in ALLOWED_DOMAINS:
        dl = d.lower()
        if cl in dl or dl in cl:
            return d
    # Keyword heuristic
    keyword_map = {
        "Machine Learning": ("learning", "ml", "supervised", "unsupervised"),
        "Deep Learning": ("neural", "deep", "transformer", "cnn", "rnn"),
        "Natural Language Processing": ("nlp", "language", "text", "token"),
        "Computer Vision": ("vision", "image", "video", "segmentation"),
        "Reinforcement Learning": ("reinforcement", "policy", "reward"),
        "Cybersecurity": ("security", "cyber", "attack", "intrusion", "malware",
                          "encryption", "privacy", "trust"),
        "Blockchain": ("blockchain", "ledger", "smart contract", "consensus"),
        "Healthcare": ("medical", "clinical", "patient", "disease", "biomedical",
                       "cancer", "diagnosis"),
        "Bioinformatics": ("genome", "protein", "rna", "dna", "gene"),
        "Robotics": ("robot", "manipulation", "navigation", "actuator"),
    }
    for d, kws in keyword_map.items():
        if any(kw in cl for kw in kws):
            return d
    return "General Research"


async def concept_extractor_node(state: ResearchState, config: RunnableConfig) -> dict:
    cfg = config.get("configurable", {})
    session = cfg.get("session")
    embedder = cfg.get("embedder")
    llm_provider = cfg.get("llm_provider")

    graph_id: str = state["graph_id"]
    document_ids: list[str] = state["document_ids"]

    if not session or not llm_provider:
        return _error_return("concept_extractor", "Missing session or llm_provider in config")

    # ── 1. Fetch chunks (DB-portable: works on PG and SQLite) ─────────────────
    try:
        # Use IN expansion for portability — ANY(:doc_ids) is asyncpg-only.
        from sqlalchemy import select as sa_select

        from apps.api.models.chunk import DocumentChunk

        # Per-document fair share so we don't burn the entire budget on one paper.
        per_doc_budget = max(8, _MAX_CHUNKS_TOTAL // max(1, len(document_ids)))
        chunks: list[dict] = []
        for doc_id in document_ids:
            rows = await session.execute(
                sa_select(
                    DocumentChunk.id,
                    DocumentChunk.document_id,
                    DocumentChunk.content,
                    DocumentChunk.section,
                    DocumentChunk.chunk_index,
                )
                .where(DocumentChunk.document_id == UUID(doc_id))
                .order_by(DocumentChunk.chunk_index)
                .limit(per_doc_budget)
            )
            for r in rows.fetchall():
                chunks.append({
                    "chunk_id": str(r.id),
                    "document_id": str(r.document_id),
                    "content": r.content or "",
                    "section": r.section or "body",
                })
            if len(chunks) >= _MAX_CHUNKS_TOTAL:
                break
    except Exception as exc:
        logger.error("concept_extractor: chunk fetch failed — %s", exc)
        return _error_return("concept_extractor", str(exc))

    # ── 1b. Drop near-duplicate chunks ────────────────────────────────────────
    # PDF extractors often produce the same conference-header / page-footer text
    # repeatedly (e.g. "©2025 IEEE | DOI: ..."). When the same boilerplate is
    # cited as evidence multiple times, the inspector shows visually-duplicate
    # quotes. We dedupe by a normalised prefix fingerprint (first 120 chars,
    # lowercased, whitespace-collapsed) per document.
    chunks = _dedupe_near_duplicates(chunks)

    if not chunks:
        return {
            "concepts": [],
            "completed_steps": ["concept_extractor"],
            "stream_events": [
                _stream_event(
                    "step_complete",
                    {"concepts_found": 0, "reason": "no_chunks"},
                )
            ],
        }

    # ── 2. Fetch document titles for hub-node creation ─────────────────────────
    doc_titles = await _fetch_document_titles(session, document_ids)

    # ── 3. LLM extraction in batches ───────────────────────────────────────────
    loop = asyncio.get_event_loop()
    extracted: list[dict] = []   # accumulates per-chunk concepts

    domain_list_str = "\n".join(f"- {d}" for d in ALLOWED_DOMAINS)

    for i in range(0, len(chunks), _CHUNKS_PER_BATCH):
        batch = chunks[i : i + _CHUNKS_PER_BATCH]
        combined = "\n\n---\n\n".join(
            f"[{c['section']}]\n{c['content'][:_CHUNK_CHARS_PER_BATCH]}" for c in batch
        )
        chunk_ids = [c["chunk_id"] for c in batch]
        doc_id_for_batch = batch[0]["document_id"] if batch else None

        prompt = _EXTRACTION_PROMPT_TEMPLATE.format(
            domains=domain_list_str, text=combined,
        )

        try:
            result = await loop.run_in_executor(
                None,
                lambda p=prompt: llm_provider.complete_json(p, max_tokens=900),
            )
            raw_concepts = result.get("concepts") or []
            if not raw_concepts and "raw" in result:
                # Last-ditch repair: scan raw text for label/domain pairs
                raw_concepts = _salvage_from_raw(str(result.get("raw", "")))

            for c in raw_concepts:
                if not isinstance(c, dict):
                    continue
                label = _normalize_label(str(c.get("label", "")))
                if not label or _is_low_value(label):
                    continue
                domain = _normalize_domain(str(c.get("domain", "")))
                try:
                    confidence = float(c.get("confidence", 0.7))
                except (TypeError, ValueError):
                    confidence = 0.7
                confidence = min(max(confidence, 0.0), 1.0)

                extracted.append({
                    "label": label,
                    "domain": domain,
                    "confidence": confidence,
                    "source_chunk_ids": chunk_ids[:5],
                    "source_document_id": doc_id_for_batch,
                })
        except Exception as exc:
            logger.warning(
                "concept_extractor: LLM batch %d failed — %s",
                i // _CHUNKS_PER_BATCH, exc,
            )

    # ── 4. Deduplicate by canonical label ──────────────────────────────────────
    by_canonical: dict[str, dict] = {}
    for c in extracted:
        key = c["label"].lower()
        existing = by_canonical.get(key)
        if existing is None:
            by_canonical[key] = {
                **c,
                "source_document_ids": [c["source_document_id"]] if c["source_document_id"] else [],
            }
        else:
            # Merge: union sources, keep highest confidence, prefer same domain.
            existing["confidence"] = max(existing["confidence"], c["confidence"])
            sd = c["source_document_id"]
            if sd and sd not in existing["source_document_ids"]:
                existing["source_document_ids"].append(sd)
            for ch in c["source_chunk_ids"]:
                if ch not in existing["source_chunk_ids"]:
                    existing["source_chunk_ids"].append(ch)

    unique = list(by_canonical.values())

    if not unique:
        return {
            "concepts": [],
            "completed_steps": ["concept_extractor"],
            "stream_events": [
                _stream_event(
                    "step_complete",
                    {"concepts_found": 0, "reason": "no_concepts_extracted"},
                )
            ],
        }

    # ── 5. Embed labels (also enables cross-doc dedup of near-duplicates) ──────
    labels = [c["label"] for c in unique]
    embeddings: list[list[float] | None] = [None] * len(unique)
    if embedder is not None:
        try:
            embs = await loop.run_in_executor(None, embedder.embed_batch, labels)
            embeddings = list(embs)
        except Exception as exc:
            logger.warning("concept_extractor: embedding failed — %s", exc)

    # Cross-document near-duplicate merging (e.g. "Convolutional Neural Network"
    # vs "Convolutional Neural Networks" or "BERT model" vs "BERT").
    unique, embeddings = _embedding_dedup(unique, embeddings, threshold=_DEDUP_SIMILARITY)

    # ── 6. Persist domain hubs, document nodes, concept nodes, edges ───────────
    domain_to_node_id: dict[str, str] = {}
    document_to_node_id: dict[str, str] = {}
    concept_nodes: list[ConceptNode] = []
    node_embedding_pairs: list[tuple[str, list[float]]] = []

    domains_seen = sorted({c["domain"] for c in unique})
    for d in domains_seen:
        node_id = uuid4()
        domain_to_node_id[d] = str(node_id)
        session.add(
            Node(
                id=node_id,
                graph_id=UUID(graph_id),
                label=d,
                node_type="domain",
                tier=1,
                cluster_id=d,
                metadata_json={"role": "domain_hub"},
            )
        )

    # Document hubs (one per source document that contributed a concept)
    contributing_docs = {
        doc_id
        for c in unique
        for doc_id in c.get("source_document_ids", [])
        if doc_id
    }
    for doc_id in sorted(contributing_docs):
        node_id = uuid4()
        document_to_node_id[doc_id] = str(node_id)
        title = doc_titles.get(doc_id, doc_id[:8])
        session.add(
            Node(
                id=node_id,
                graph_id=UUID(graph_id),
                document_id=UUID(doc_id),
                label=title[:500],
                node_type="document",
                tier=2,
                cluster_id="Source Documents",
                metadata_json={"role": "document_hub", "source_document_id": doc_id},
            )
        )

    # Concept nodes + edges
    for concept, embedding in zip(unique, embeddings):
        node_id = uuid4()
        session.add(
            Node(
                id=node_id,
                graph_id=UUID(graph_id),
                label=concept["label"][:500],
                node_type="concept",
                tier=3,
                cluster_id=concept["domain"][:100],
                metadata_json={
                    "domain": concept["domain"],
                    "confidence": concept["confidence"],
                    "extracted_by": "concept_extractor",
                    "source_chunk_ids": concept["source_chunk_ids"][:8],
                    "source_document_ids": concept.get("source_document_ids", []),
                    # Store embedding inline for SQLite/non-pgvector environments
                    # so the bridge agent can do Python-side cosine similarity.
                    "embedding_inline": embedding if embedding is not None else None,
                },
            )
        )

        if embedding is not None:
            node_embedding_pairs.append((str(node_id), embedding))

        # has_concept edge: domain hub → concept
        domain_node_id = domain_to_node_id.get(concept["domain"])
        if domain_node_id is not None:
            session.add(
                EdgeModel(
                    id=uuid4(),
                    graph_id=UUID(graph_id),
                    source_node_id=UUID(domain_node_id),
                    target_node_id=node_id,
                    edge_type="has_concept",
                    edge_category="HIERARCHICAL",
                    bridge_concept=None,
                    evidence=f"{concept['domain']} → {concept['label']}",
                    reasoning_trace_id=uuid4(),
                    confidence=concept["confidence"],
                    metadata_json={"role": "domain_concept_link"},
                )
            )

        # extracted_from edge: concept → each contributing document hub
        for doc_id in concept.get("source_document_ids", []):
            doc_node_id = document_to_node_id.get(doc_id)
            if doc_node_id is None:
                continue
            session.add(
                EdgeModel(
                    id=uuid4(),
                    graph_id=UUID(graph_id),
                    source_node_id=node_id,
                    target_node_id=UUID(doc_node_id),
                    edge_type="extracted_from",
                    edge_category="PROVENANCE",
                    bridge_concept=None,
                    evidence=(
                        f"Extracted '{concept['label']}' from "
                        f"{doc_titles.get(doc_id, doc_id[:8])}"
                    ),
                    reasoning_trace_id=uuid4(),
                    confidence=concept["confidence"],
                    metadata_json={"role": "extraction_provenance", "document_id": doc_id},
                )
            )

        concept_nodes.append(
            ConceptNode(
                node_id=str(node_id),
                label=concept["label"],
                domain=concept["domain"],
                source_chunk_ids=concept["source_chunk_ids"],
                confidence=concept["confidence"],
            )
        )

    await session.flush()

    # ── 7. Update pgvector embeddings (no-op on SQLite) ───────────────────────
    vs = VectorStore(session)
    emb_ok = 0
    for node_id_str, embedding in node_embedding_pairs:
        try:
            async with session.begin_nested():
                await vs.update_node_embedding(node_id_str, embedding)
            emb_ok += 1
        except Exception as exc:
            logger.warning(
                "concept_extractor: embedding update skipped for %s — %s", node_id_str, exc
            )
    logger.info(
        "concept_extractor: %d concepts extracted (across %d docs, %d domains); "
        "wrote pgvector embeddings for %d/%d nodes",
        len(concept_nodes), len(contributing_docs), len(domains_seen),
        emb_ok, len(node_embedding_pairs),
    )

    event = _stream_event(
        "step_complete",
        {
            "concepts_found": len(concept_nodes),
            "domains_found": len(domains_seen),
            "documents": len(contributing_docs),
            "chunks_processed": len(chunks),
        },
    )
    return {
        "concepts": concept_nodes,
        "completed_steps": ["concept_extractor"],
        "stream_events": [event],
    }


async def _fetch_document_titles(session, document_ids: list[str]) -> dict[str, str]:
    """Best-effort title lookup; falls back to filename or the doc id prefix."""
    try:
        from sqlalchemy import select as sa_select

        from apps.api.models.document import Document
        rows = await session.execute(
            sa_select(Document.id, Document.filename, Document.metadata_json)
            .where(Document.id.in_([UUID(d) for d in document_ids]))
        )
        out: dict[str, str] = {}
        for r in rows.fetchall():
            meta = r.metadata_json or {}
            title = (
                str(meta.get("title") or "").strip()
                or (r.filename or "").strip()
                or str(r.id)[:8]
            )
            out[str(r.id)] = title
        return out
    except Exception as exc:
        logger.debug("concept_extractor: doc title fetch failed — %s", exc)
        return {}


def _dedupe_near_duplicates(chunks: list[dict]) -> list[dict]:
    """Drop chunks whose normalised prefix matches one already kept.

    Operates per-document so the same paragraph can still be cited from two
    different papers. Preserves original ordering and chunk_id references.
    """
    seen: dict[str, set[str]] = {}
    kept: list[dict] = []
    for c in chunks:
        doc_id = c.get("document_id", "")
        content = (c.get("content") or "").strip()
        if not content:
            continue
        # Normalise: lowercase + collapse whitespace + take first 120 chars.
        normalised = re.sub(r"\s+", " ", content.lower())[:120]
        bucket = seen.setdefault(doc_id, set())
        if normalised in bucket:
            continue
        bucket.add(normalised)
        kept.append(c)
    return kept


def _embedding_dedup(
    concepts: list[dict],
    embeddings: list[list[float] | None],
    *,
    threshold: float,
) -> tuple[list[dict], list[list[float] | None]]:
    """Greedy embedding-based dedup: merge concepts whose vectors are very close."""
    if not concepts:
        return concepts, embeddings

    keep_idx: list[int] = []
    # Parallel array: concept i was merged into keep_idx[canonical_for[i]]
    canonical_for: list[int] = []

    for i, emb_i in enumerate(embeddings):
        merged_into: int | None = None
        if emb_i is not None:
            for k in keep_idx:
                emb_k = embeddings[k]
                if emb_k is None:
                    continue
                if _cosine(emb_i, emb_k) >= threshold:
                    merged_into = k
                    break
        if merged_into is None:
            keep_idx.append(i)
            canonical_for.append(i)
        else:
            canonical_for.append(merged_into)
            # Merge metadata into the canonical concept
            target = concepts[merged_into]
            src = concepts[i]
            target["confidence"] = max(target["confidence"], src["confidence"])
            for sd in src.get("source_document_ids", []):
                if sd not in target.setdefault("source_document_ids", []):
                    target["source_document_ids"].append(sd)
            for ch in src.get("source_chunk_ids", []):
                if ch not in target["source_chunk_ids"]:
                    target["source_chunk_ids"].append(ch)

    deduped_concepts = [concepts[i] for i in keep_idx]
    deduped_embeddings = [embeddings[i] for i in keep_idx]
    return deduped_concepts, deduped_embeddings


def _cosine(a: list[float], b: list[float]) -> float:
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


def _salvage_from_raw(raw: str) -> list[dict]:
    """Try to recover a concepts array from a malformed LLM response."""
    if not raw:
        return []
    # Look for any embedded JSON list of concept dicts
    match = re.search(r'"concepts"\s*:\s*(\[[\s\S]*?\])', raw)
    if match:
        try:
            return json.loads(match.group(1))
        except json.JSONDecodeError:
            pass
    return []


def _error_return(node: str, msg: str) -> dict:
    return {
        "concepts": [],
        "completed_steps": [node],
        "error": msg,
        "stream_events": [{
            "agent_node": node,
            "event_type": "error",
            "content": {"error": msg},
            "timestamp": _now_iso(),
        }],
    }
