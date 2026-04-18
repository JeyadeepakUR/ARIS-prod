"""Graph orchestration service for ARIS knowledge graph pipeline."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
import re
from collections import Counter
from itertools import combinations
from uuid import UUID, uuid4

from sqlalchemy.ext.asyncio import AsyncSession

from apps.api.config import get_settings
from apps.api.models.document import Document
from apps.api.models.edge import Edge as EdgeModel
from apps.api.models.node import Node as NodeModel
from aris.core.evaluation import Evaluator
from aris.core.reasoning_engine import ReasoningEngine
from aris.graph.document_ingestion import Document as CoreDocument
from aris.graph.document_ingestion import create_corpus
from aris.graph.knowledge_graph import Edge
from aris.graph.knowledge_graph import KnowledgeGraph
from aris.graph.knowledge_graph import Linker, LinkMaterializer, Node
from aris.graph.knowledge_graph import add_edges_to_graph, build_graph_from_corpus
from aris.graph.research_planner import PlannerContext, ResearchAction, ResearchPlanner
from aris.llm.provider import LLMProvider, get_provider


@dataclass(frozen=True)
class GraphBuildArtifacts:
    """In-memory outputs from a full graph build run."""

    graph: KnowledgeGraph
    plan_actions: list[ResearchAction]
    trace: dict[str, object]


class GraphService:
    """Build knowledge graph and draft plans from ingested workspace documents."""

    # Seed keyword vocabulary for domain detection (supplemented by LLM when available)
    _DOMAIN_KEYWORDS: dict[str, set[str]] = {
        "machine_learning": {
            "ml", "ai", "model", "neural", "learning", "classification",
            "prediction", "inference", "deep", "transformer", "embedding",
            "training", "dataset", "attention", "fine-tuning", "pretraining",
        },
        "cybersecurity": {
            "security", "cyber", "threat", "attack", "malware", "anomaly",
            "intrusion", "encryption", "privacy", "forensics", "audit", "trust",
            "vulnerability", "exploit", "phishing", "authentication",
        },
        "blockchain": {
            "blockchain", "ledger", "smart", "contract", "consensus", "token",
            "immutable", "decentralized", "proof", "cryptocurrency", "defi",
        },
        "healthcare": {
            "cancer", "clinical", "diagnosis", "patient", "medical", "biometric",
            "depression", "therapy", "detection", "drug", "genomics", "radiology",
            "ehr", "epidemiology", "biomarker",
        },
        "computer_vision": {
            "image", "vision", "video", "spectrogram", "detection", "segmentation",
            "recognition", "captioning", "object", "pixel", "convolution", "vit",
        },
        "natural_language_processing": {
            "nlp", "text", "language", "sentiment", "parsing", "tokenization",
            "summarization", "translation", "question", "answering", "dialogue",
        },
        "robotics": {
            "robot", "autonomous", "navigation", "manipulation", "control",
            "reinforcement", "planning", "sensor", "actuator", "localization",
        },
        "bioinformatics": {
            "genome", "protein", "sequence", "alignment", "phylogenetic",
            "rna", "dna", "gene", "expression", "mutation", "variant",
        },
    }

    def __init__(self, provider: LLMProvider | None = None) -> None:
        if provider is None:
            settings = get_settings()
            try:
                provider = get_provider(
                    settings.llm_provider,
                    model=settings.llm_model,
                    api_key=settings.llm_api_key,
                    base_url=settings.llm_base_url,
                )
            except Exception:
                from aris.llm.mock_provider import MockProvider
                provider = MockProvider()
        self._provider = provider

    _STOPWORDS: set[str] = {
        "the",
        "and",
        "for",
        "with",
        "from",
        "that",
        "this",
        "into",
        "using",
        "based",
        "study",
        "system",
        "analysis",
        "paper",
        "research",
        "through",
        "between",
        "over",
        "under",
        "data",
        "method",
    }

    def build_graph(
        self,
        documents: list[Document],
        *,
        strategy: str,
        keyword: str | None,
        metadata_field: str | None,
        metadata_value: str | None,
        plan_strategy: str,
        max_plan_actions: int,
    ) -> GraphBuildArtifacts:
        core_documents = [self._to_core_document(document) for document in documents]

        normalized_strategy = strategy.strip().lower()
        if normalized_strategy in {"domain_network", "domain-bridge-network", "network"}:
            network_graph, network_trace = self._build_domain_network(core_documents)
            planner = ResearchPlanner()
            plan_actions = planner.plan(
                PlannerContext(
                    graph=network_graph,
                    strategy=plan_strategy,
                    max_actions=max_plan_actions,
                )
            )
            trace = {
                "strategy": "domain_network",
                **network_trace,
                "plan_strategy": plan_strategy,
                "plan_actions_count": len(plan_actions),
            }
            return GraphBuildArtifacts(graph=network_graph, plan_actions=plan_actions, trace=trace)

        corpus = create_corpus(core_documents)
        base_graph = build_graph_from_corpus(corpus)

        linker = Linker()
        candidates = self._generate_candidates(
            linker=linker,
            corpus=corpus,
            strategy=strategy,
            keyword=keyword,
            metadata_field=metadata_field,
            metadata_value=metadata_value,
        )

        materializer = LinkMaterializer(ReasoningEngine(self._provider), Evaluator())
        edges, failures, rejections = materializer.materialize_batch(candidates, corpus)
        if not edges:
            edges = self._fallback_edges(core_documents)

        graph = add_edges_to_graph(base_graph, edges)

        planner = ResearchPlanner()
        plan_actions = planner.plan(
            PlannerContext(
                graph=graph,
                strategy=plan_strategy,
                max_actions=max_plan_actions,
            )
        )

        trace = {
            "strategy": strategy,
            "candidate_count": len(candidates),
            "edge_count": len(edges),
            "materialization_failures": [str(exc) for _, exc in failures],
            "materialization_rejections": len(rejections),
            "plan_strategy": plan_strategy,
            "plan_actions_count": len(plan_actions),
        }

        return GraphBuildArtifacts(graph=graph, plan_actions=plan_actions, trace=trace)

    def create_plans_from_persisted(
        self,
        *,
        graph_id: UUID,
        nodes: list[NodeModel],
        edges: list[EdgeModel],
        strategy: str,
        max_actions: int,
    ) -> list[ResearchAction]:
        planner = ResearchPlanner()
        graph = KnowledgeGraph(
            graph_id=graph_id,
            nodes=[
                Node(
                    node_id=node.id,
                    document_id=node.document_id or node.id,
                    label=node.label,
                    node_type=node.node_type,
                    metadata={k: str(v) for k, v in node.metadata_json.items()},
                )
                for node in nodes
            ],
            edges=[
                Edge(
                    edge_id=edge.id,
                    source_id=edge.source_node_id,
                    target_id=edge.target_node_id,
                    edge_type=edge.edge_type,
                    evidence=edge.evidence,
                    reasoning_trace_id=edge.reasoning_trace_id,
                    confidence=edge.confidence,
                    metadata={k: str(v) for k, v in edge.metadata_json.items()},
                    created_at=edge.created_at,
                )
                for edge in edges
            ],
            created_at=datetime.now(UTC),
        )
        return planner.plan(PlannerContext(graph=graph, strategy=strategy, max_actions=max_actions))

    async def classify_nodes_and_edges(
        self,
        graph_id: UUID,
        nodes: list[NodeModel],
        edges: list[EdgeModel],
        db: AsyncSession,
    ) -> None:
        """Classify node tiers/clusters and edge categories after graph materialization."""

        nodes_by_id = {node.id: node for node in nodes}

        for node in nodes:
            meta = node.metadata_json if isinstance(node.metadata_json, dict) else {}

            # For domain_network graphs, metadata already carries the correct domain.
            # Prefer that over LLM classification to avoid cross-contamination.
            if node.node_type == "domain":
                node.tier = 1
                node.cluster_id = node.label  # e.g. "Machine Learning"
            elif node.node_type in {"concept", "bridge_concept"} and "domain" in meta:
                raw_domain = str(meta["domain"])
                node.tier = 3 if node.node_type == "concept" else 2
                node.cluster_id = raw_domain.replace("_", " ").title()
            elif node.node_type == "document":
                node.tier = 1
                node.cluster_id = self._derive_cluster(node.label)
            else:
                source_title = str(meta.get("source", ""))
                result = self._classify_node_with_llm_fallback(node.label, source_title)
                node.tier = int(result["tier"])
                cluster = result["cluster_id"]
                node.cluster_id = cluster if isinstance(cluster, str) and cluster else self._derive_cluster(node.label)
                if bool(result["low_value"]):
                    node.metadata_json = {**meta, "low_value": True}

        for edge in edges:
            meta = edge.metadata_json if isinstance(edge.metadata_json, dict) else {}

            # cross_domain_bridge edges are always inter-domain by construction.
            if edge.edge_type == "cross_domain_bridge":
                edge.edge_category = "INTER_DOMAIN_BRIDGE"
                if not edge.bridge_concept:
                    edge.bridge_concept = str(meta.get("bridge_concept", "")) or None
                continue

            # belongs_to_domain and has_concept are always intra-domain.
            if edge.edge_type in {"belongs_to_domain", "has_concept"}:
                edge.edge_category = "INTRA_DOMAIN"
                continue

            source_node = nodes_by_id.get(edge.source_node_id)
            target_node = nodes_by_id.get(edge.target_node_id)
            if source_node is None or target_node is None:
                edge.edge_category = "INTRA_DOMAIN"
                continue

            src_cluster = source_node.cluster_id or ""
            tgt_cluster = target_node.cluster_id or ""

            if src_cluster != tgt_cluster and src_cluster and tgt_cluster:
                edge.edge_category = "INTER_DOMAIN_BRIDGE"
                if not edge.bridge_concept:
                    edge.bridge_concept = self._synthesize_bridge_concept_with_llm_fallback(
                        source_node, target_node, edge
                    )
            elif abs(source_node.tier - target_node.tier) == 1 and src_cluster == tgt_cluster:
                edge.edge_category = "HIERARCHICAL"
            else:
                edge.edge_category = "INTRA_DOMAIN"

        await db.flush()

    def _classify_node_with_llm_fallback(self, label: str, source_title: str) -> dict[str, object]:
        try:
            prompt = (
                "Classify this knowledge graph node for a research document.\n\n"
                f"Node label: {label}\n"
                f"Source document: {source_title}\n\n"
                "Return a JSON object with:\n"
                '- "tier": integer 1 (broad domain), 2 (sub-domain/technique), or 3 (specific concept/term)\n'
                '- "cluster_id": short domain name string (e.g. "Machine Learning", "Cybersecurity")\n'
                '- "low_value": boolean — true only for extremely generic terms like "method", "system"\n\n'
                "Return only the JSON object."
            )
            result = self._provider.complete_json(prompt, max_tokens=120)
            tier = int(result.get("tier", 3))
            cluster = str(result.get("cluster_id", "") or "")
            low_value = bool(result.get("low_value", False))
            if tier not in {1, 2, 3}:
                tier = 3
            if not cluster:
                cluster = self._derive_cluster(f"{label} {source_title}")
            return {"tier": tier, "cluster_id": cluster, "low_value": low_value}
        except Exception:
            return self._classify_node_heuristic(label, source_title)

    def _classify_node_heuristic(self, label: str, source_title: str) -> dict[str, object]:
        normalized_label = label.strip().lower()
        low_value_terms = {
            "learning", "model", "system", "method", "approach", "result",
            "performance", "technique", "application", "using", "based",
            "analysis", "framework",
        }
        if normalized_label in low_value_terms:
            return {"tier": 3, "cluster_id": self._derive_cluster(source_title or label), "low_value": True}
        if any(token in normalized_label for token in ["domain", "field", "research area"]):
            return {"tier": 1, "cluster_id": self._derive_cluster(label), "low_value": False}
        if any(token in normalized_label for token in ["detection", "learning", "security", "contracts"]):
            return {"tier": 2, "cluster_id": self._derive_cluster(label), "low_value": False}
        return {"tier": 3, "cluster_id": self._derive_cluster(f"{label} {source_title}"), "low_value": False}

    def _derive_cluster(self, text: str) -> str:
        lowered = text.lower()
        for key, keywords in self._DOMAIN_KEYWORDS.items():
            if any(keyword in lowered for keyword in keywords):
                return key.replace("_", " ").title()
        return "General Research"

    def _synthesize_bridge_concept_with_llm_fallback(
        self,
        source_node: NodeModel,
        target_node: NodeModel,
        edge: EdgeModel,
    ) -> str:
        source_cluster = source_node.cluster_id or "general"
        target_cluster = target_node.cluster_id or "general"
        evidence_snippet = edge.evidence[:400] if edge.evidence else ""

        try:
            prompt = (
                "You are identifying cross-domain bridge concepts in a research knowledge graph.\n\n"
                f"Source domain cluster: {source_cluster}\n"
                f"Source node: {source_node.label}\n"
                f"Target domain cluster: {target_cluster}\n"
                f"Target node: {target_node.label}\n"
                f"Edge evidence: {evidence_snippet}\n\n"
                "Identify the precise conceptual bridge that connects these two domains. "
                "The bridge concept should be a specific mechanism, formalism, technique, or principle "
                "that is genuinely shared between both domains and enables knowledge transfer.\n\n"
                "Return a JSON object with:\n"
                '- "bridge_concept": concise name (3-8 words) of the bridge concept\n'
                '- "description": one sentence explaining the transfer mechanism\n\n'
                "Return only the JSON object."
            )
            result = self._provider.complete_json(prompt, max_tokens=200)
            bridge = str(result.get("bridge_concept", "")).strip()
            if bridge:
                return bridge
        except Exception:
            pass

        # Heuristic fallback
        evidence_words = [
            token for token in re.findall(r"[a-z]{4,}", edge.evidence.lower())
            if token not in self._STOPWORDS
        ]
        keyword = evidence_words[0] if evidence_words else "integration"
        return f"{source_cluster} {target_cluster} {keyword}".strip()

    def _to_core_document(self, document: Document) -> CoreDocument:
        metadata = {k: str(v) for k, v in document.metadata_json.items()}
        content = self._extract_content(document, metadata)
        return CoreDocument(
            content=content,
            source=document.filename,
            format=document.file_format,
            metadata=metadata,
            document_id=document.id,
            created_at=datetime.now(UTC),
        )

    def _extract_content(self, document: Document, metadata: dict[str, str]) -> str:
        """Return the best available document text for graph building."""
        if document.full_text:
            return document.full_text

        # Fall back to preview stored in metadata (legacy documents)
        preview = metadata.get("content_preview", "")
        if preview:
            return re.sub(r"\s+", " ", preview).strip()

        # Last resort: synthesize from document metadata
        source = metadata.get("source", document.filename)
        return (
            f"Document {document.filename} format={document.file_format} "
            f"workspace={document.workspace_id} source={source}"
        )

    def _generate_candidates(
        self,
        *,
        linker: Linker,
        corpus,
        strategy: str,
        keyword: str | None,
        metadata_field: str | None,
        metadata_value: str | None,
    ):
        normalized_strategy = strategy.strip().lower()
        try:
            return linker.generate_candidates(
                corpus,
                strategy=normalized_strategy,
                keyword=(keyword or "document").lower(),
                metadata_field=metadata_field or "source",
                metadata_value=metadata_value or "",
            )
        except ValueError:
            return linker.generate_candidates(corpus, strategy="sequential")

    def _fallback_edges(self, documents: list[CoreDocument]) -> list[Edge]:
        fallback_edges: list[Edge] = []
        for index in range(len(documents) - 1):
            source_doc = documents[index]
            target_doc = documents[index + 1]
            fallback_edges.append(
                Edge(
                    edge_id=uuid4(),
                    source_id=source_doc.document_id,
                    target_id=target_doc.document_id,
                    edge_type="sequential",
                    evidence=f"Fallback sequential link between {source_doc.source} and {target_doc.source}",
                    reasoning_trace_id=uuid4(),
                    confidence=0.7,
                    metadata={"strategy": "fallback_sequential"},
                    created_at=datetime.now(UTC),
                )
            )

        return fallback_edges

    def _build_domain_network(self, documents: list[CoreDocument]) -> tuple[KnowledgeGraph, dict[str, object]]:
        nodes: list[Node] = []
        edges: list[Edge] = []
        bridge_count = 0
        domain_count = 0
        concept_count = 0

        # ── Shared concept registry: concept_label → Node (dedup across docs) ──
        # Same concept appearing in multiple documents shares ONE node.
        shared_concept_nodes: dict[str, Node] = {}   # label → Node
        # domain_label → Node (also deduped: same domain across docs = one node)
        shared_domain_nodes: dict[str, Node] = {}

        for document in documents:
            doc_node = Node(
                node_id=uuid4(),
                document_id=document.document_id,
                label=document.source,
                node_type="document",
                metadata={"format": document.format},
            )
            nodes.append(doc_node)

            # Use LLM to extract meaningful named concepts (not raw keyword counts)
            domain_to_concepts = self._extract_document_concepts_with_llm(
                document.content, document.source
            )

            if len(domain_to_concepts) < 2:
                continue

            domain_nodes: dict[str, Node] = {}

            for domain, concepts in domain_to_concepts.items():
                domain_label = domain.replace("_", " ").title()

                # Reuse existing domain node if this domain already seen across docs
                if domain_label in shared_domain_nodes:
                    d_node = shared_domain_nodes[domain_label]
                else:
                    d_node = Node(
                        node_id=uuid4(),
                        document_id=document.document_id,
                        label=domain_label,
                        node_type="domain",
                        metadata={"domain": domain, "document": document.source},
                    )
                    shared_domain_nodes[domain_label] = d_node
                    nodes.append(d_node)
                    domain_count += 1

                domain_nodes[domain] = d_node

                edges.append(
                    Edge(
                        edge_id=uuid4(),
                        source_id=doc_node.node_id,
                        target_id=d_node.node_id,
                        edge_type="belongs_to_domain",
                        evidence=f"Document '{document.source}' contains research in {domain_label}.",
                        reasoning_trace_id=uuid4(),
                        confidence=0.82,
                        metadata={"strategy": "domain_network", "domain": domain},
                        created_at=datetime.now(UTC),
                    )
                )

                for concept in concepts:
                    concept_label = concept.strip().lower()
                    if not concept_label:
                        continue

                    # Reuse node if this concept already exists (from another doc or domain)
                    if concept_label in shared_concept_nodes:
                        c_node = shared_concept_nodes[concept_label]
                    else:
                        c_node = Node(
                            node_id=uuid4(),
                            document_id=document.document_id,
                            label=concept_label,
                            node_type="concept",
                            metadata={"domain": domain, "document": document.source},
                        )
                        shared_concept_nodes[concept_label] = c_node
                        nodes.append(c_node)
                        concept_count += 1

                    # Always add the domain→concept edge (even for reused nodes)
                    edges.append(
                        Edge(
                            edge_id=uuid4(),
                            source_id=d_node.node_id,
                            target_id=c_node.node_id,
                            edge_type="has_concept",
                            evidence=(
                                f"'{concept_label}' is a key concept in {domain_label} "
                                f"as identified in '{document.source}'."
                            ),
                            reasoning_trace_id=uuid4(),
                            confidence=0.8,
                            metadata={"strategy": "domain_network", "domain": domain, "concept": concept_label},
                            created_at=datetime.now(UTC),
                        )
                    )

            # ── Bridge detection: concepts shared across domain pairs ──
            for left_domain, right_domain in combinations(domain_to_concepts.keys(), 2):
                left_concepts = set(c.strip().lower() for c in domain_to_concepts[left_domain])
                right_concepts = set(c.strip().lower() for c in domain_to_concepts[right_domain])
                bridges = sorted(left_concepts & right_concepts)
                if not bridges:
                    continue

                left_node = domain_nodes[left_domain]
                right_node = domain_nodes[right_domain]

                for bridge in bridges[: self._MAX_BRIDGES_PER_PAIR]:
                    # Upgrade the shared concept node to bridge_concept type
                    if bridge in shared_concept_nodes:
                        bridge_node = shared_concept_nodes[bridge]
                        bridge_node = Node(
                            node_id=bridge_node.node_id,
                            document_id=bridge_node.document_id,
                            label=bridge_node.label,
                            node_type="bridge_concept",
                            metadata={
                                "domains": f"{left_domain},{right_domain}",
                                "document": document.source,
                            },
                        )
                        # Replace in shared registry and node list
                        shared_concept_nodes[bridge] = bridge_node
                        nodes = [bridge_node if n.node_id == bridge_node.node_id else n for n in nodes]
                    else:
                        bridge_node = Node(
                            node_id=uuid4(),
                            document_id=document.document_id,
                            label=bridge,
                            node_type="bridge_concept",
                            metadata={
                                "domains": f"{left_domain},{right_domain}",
                                "document": document.source,
                            },
                        )
                        shared_concept_nodes[bridge] = bridge_node
                        nodes.append(bridge_node)

                    bridge_count += 1
                    bridge_meta = {
                        "strategy": "domain_network",
                        "bridge_concept": bridge,
                        "from": left_domain,
                        "to": right_domain,
                    }
                    bridge_evidence = (
                        f"'{bridge}' bridges {left_domain.replace('_',' ')} and "
                        f"{right_domain.replace('_',' ')} in '{document.source}'."
                    )
                    edges.append(Edge(
                        edge_id=uuid4(),
                        source_id=left_node.node_id,
                        target_id=bridge_node.node_id,
                        edge_type="cross_domain_bridge",
                        evidence=bridge_evidence,
                        reasoning_trace_id=uuid4(),
                        confidence=0.86,
                        metadata=bridge_meta,
                        created_at=datetime.now(UTC),
                    ))
                    edges.append(Edge(
                        edge_id=uuid4(),
                        source_id=bridge_node.node_id,
                        target_id=right_node.node_id,
                        edge_type="cross_domain_bridge",
                        evidence=bridge_evidence,
                        reasoning_trace_id=uuid4(),
                        confidence=0.86,
                        metadata=bridge_meta,
                        created_at=datetime.now(UTC),
                    ))

        graph = KnowledgeGraph(
            graph_id=uuid4(),
            nodes=nodes,
            edges=edges,
            created_at=datetime.now(UTC),
        )

        trace = {
            "documents_count": len(documents),
            "domains_count": domain_count,
            "concepts_count": concept_count,
            "bridges_count": bridge_count,
            "nodes_count": len(nodes),
            "edges_count": len(edges),
        }
        return graph, trace

    # ------------------------------------------------------------------
    # LLM-based concept extraction (replaces keyword-frequency approach)
    # ------------------------------------------------------------------

    _MAX_DOMAINS_PER_DOC = 4
    _MAX_CONCEPTS_PER_DOMAIN = 6
    _MAX_BRIDGES_PER_PAIR = 3

    def _extract_document_concepts_with_llm(self, content: str, source: str) -> dict[str, list[str]]:
        """
        Ask the LLM to identify the key research domains and their most important
        named concepts from this document.  Returns {domain_key: [concept, ...]}
        with at most _MAX_DOMAINS_PER_DOC domains and _MAX_CONCEPTS_PER_DOMAIN
        concepts each.
        """
        excerpt = content[:3000]
        try:
            prompt = (
                "You are a research knowledge graph builder. "
                "Analyse the following excerpt from a research paper and extract:\n"
                "1. The 2-4 primary research DOMAINS (e.g. 'machine_learning', 'computer_vision', "
                "'natural_language_processing', 'cybersecurity', 'robotics', 'healthcare', 'blockchain').\n"
                "2. For each domain, the 4-6 most important NAMED research concepts — these must be "
                "specific multi-word or compound terms actually used in the paper "
                "(e.g. 'transformer architecture', 'contrastive learning', 'federated learning', "
                "'image captioning', 'anomaly detection', 'proof-of-stake consensus'). "
                "Do NOT use generic single words like 'model', 'data', 'deep', 'learning', 'image'.\n\n"
                f"Paper excerpt:\n{excerpt}\n\n"
                "Return ONLY a JSON object:\n"
                '{"domains": [{"name": "domain_key", "concepts": ["concept 1", "concept 2", ...]}]}'
            )
            result = self._provider.complete_json(prompt, max_tokens=400)
            domains_raw = result.get("domains", [])
            if not isinstance(domains_raw, list):
                raise ValueError("bad shape")

            out: dict[str, list[str]] = {}
            for entry in domains_raw[: self._MAX_DOMAINS_PER_DOC]:
                domain_name = str(entry.get("name", "")).strip().lower().replace(" ", "_")
                concepts_raw = entry.get("concepts", [])
                if not domain_name or not isinstance(concepts_raw, list):
                    continue
                cleaned = []
                for c in concepts_raw:
                    c = str(c).strip().lower()
                    # Drop single bare words that are stopwords or too generic
                    if c and len(c) >= 4 and c not in self._STOPWORDS:
                        cleaned.append(c)
                if cleaned:
                    out[domain_name] = cleaned[: self._MAX_CONCEPTS_PER_DOMAIN]

            return out if len(out) >= 2 else {}
        except Exception:
            pass

        # Fallback to keyword matching if LLM fails
        return self._extract_domain_concepts_fallback(self._tokenize(content))

    def _extract_domain_concepts_fallback(self, tokens: list[str]) -> dict[str, list[str]]:
        """Keyword-frequency fallback used only when LLM is unavailable."""
        domain_scores: dict[str, int] = {}
        for domain, keywords in self._DOMAIN_KEYWORDS.items():
            score = sum(1 for token in tokens if token in keywords)
            if score >= 2:
                domain_scores[domain] = score

        if len(domain_scores) < 2:
            return {}

        token_counts = Counter(
            token for token in tokens
            if token not in self._STOPWORDS and len(token) >= 5 and not token.isdigit()
        )

        domain_to_concepts: dict[str, list[str]] = {}
        top_domains = sorted(domain_scores, key=lambda d: -domain_scores[d])[: self._MAX_DOMAINS_PER_DOC]
        for domain in top_domains:
            kw = self._DOMAIN_KEYWORDS[domain]
            domain_specific = [t for t in token_counts if t in kw]
            merged: list[str] = []
            for t in domain_specific:
                if t not in merged:
                    merged.append(t)
                if len(merged) >= self._MAX_CONCEPTS_PER_DOMAIN:
                    break
            if merged:
                domain_to_concepts[domain] = merged

        return domain_to_concepts if len(domain_to_concepts) >= 2 else {}

    def _tokenize(self, text: str) -> list[str]:
        return re.findall(r"[a-z][a-z0-9_+-]{2,}", text.lower())

    def _extract_domains_with_llm(self, content_snippet: str) -> list[str]:
        """Ask the LLM to identify research domains present in a document snippet."""
        try:
            prompt = (
                "Identify the research domains present in the following document excerpt.\n\n"
                f"Excerpt:\n{content_snippet[:1500]}\n\n"
                "Return a JSON object with:\n"
                '- "domains": list of 1-4 short domain name strings (e.g. ["machine learning", "cybersecurity"])\n'
                '- "primary_domain": the most prominent domain\n\n'
                "Return only the JSON object."
            )
            result = self._provider.complete_json(prompt, max_tokens=150)
            domains = result.get("domains", [])
            if isinstance(domains, list):
                return [str(d).strip().lower().replace(" ", "_") for d in domains if d][:4]
        except Exception:
            pass
        return []
