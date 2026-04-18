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

        for document in documents:
            doc_node = Node(
                node_id=uuid4(),
                document_id=document.document_id,
                label=document.source,
                node_type="document",
                metadata={"format": document.format},
            )
            nodes.append(doc_node)

            tokens = self._tokenize(document.content)
            domain_to_concepts = self._extract_domain_concepts(tokens)

            # Supplement with LLM-identified domains when keyword matching is insufficient
            if len(domain_to_concepts) < 2:
                llm_domains = self._extract_domains_with_llm(document.content)
                for domain in llm_domains:
                    if domain not in domain_to_concepts:
                        domain_specific = [t for t in tokens if len(t) >= 4 and t not in self._STOPWORDS]
                        top_concepts = [t for t, _ in Counter(domain_specific).most_common(6)]
                        if top_concepts:
                            domain_to_concepts[domain] = top_concepts

            if len(domain_to_concepts) < 2:
                continue

            domain_nodes: dict[str, Node] = {}
            concept_nodes: dict[tuple[str, str], Node] = {}

            for domain, concepts in domain_to_concepts.items():
                d_node = Node(
                    node_id=uuid4(),
                    document_id=document.document_id,
                    label=domain.replace("_", " ").title(),
                    node_type="domain",
                    metadata={"domain": domain, "document": document.source},
                )
                domain_nodes[domain] = d_node
                nodes.append(d_node)
                domain_count += 1

                edges.append(
                    Edge(
                        edge_id=uuid4(),
                        source_id=doc_node.node_id,
                        target_id=d_node.node_id,
                        edge_type="belongs_to_domain",
                        evidence=f"Document {document.source} contains signals for domain {domain}.",
                        reasoning_trace_id=uuid4(),
                        confidence=0.82,
                        metadata={"strategy": "domain_network", "domain": domain},
                        created_at=datetime.now(UTC),
                    )
                )

                for concept in concepts:
                    c_node = Node(
                        node_id=uuid4(),
                        document_id=document.document_id,
                        label=concept,
                        node_type="concept",
                        metadata={"domain": domain, "document": document.source},
                    )
                    concept_nodes[(domain, concept)] = c_node
                    nodes.append(c_node)
                    concept_count += 1

                    edges.append(
                        Edge(
                            edge_id=uuid4(),
                            source_id=d_node.node_id,
                            target_id=c_node.node_id,
                            edge_type="has_concept",
                            evidence=(
                                f"Concept '{concept}' is strongly associated with domain {domain} "
                                f"in {document.source}."
                            ),
                            reasoning_trace_id=uuid4(),
                            confidence=0.8,
                            metadata={"strategy": "domain_network", "domain": domain, "concept": concept},
                            created_at=datetime.now(UTC),
                        )
                    )

            for left_domain, right_domain in combinations(domain_to_concepts.keys(), 2):
                bridges = sorted(set(domain_to_concepts[left_domain]) & set(domain_to_concepts[right_domain]))
                if not bridges:
                    continue

                left_node = domain_nodes[left_domain]
                right_node = domain_nodes[right_domain]

                for bridge in bridges[:3]:
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
                    nodes.append(bridge_node)
                    bridge_count += 1

                    edges.append(
                        Edge(
                            edge_id=uuid4(),
                            source_id=left_node.node_id,
                            target_id=bridge_node.node_id,
                            edge_type="cross_domain_bridge",
                            evidence=(
                                f"Bridge concept '{bridge}' links {left_domain} and {right_domain} "
                                f"within {document.source}."
                            ),
                            reasoning_trace_id=uuid4(),
                            confidence=0.86,
                            metadata={
                                "strategy": "domain_network",
                                "bridge_concept": bridge,
                                "from": left_domain,
                                "to": right_domain,
                            },
                            created_at=datetime.now(UTC),
                        )
                    )
                    edges.append(
                        Edge(
                            edge_id=uuid4(),
                            source_id=bridge_node.node_id,
                            target_id=right_node.node_id,
                            edge_type="cross_domain_bridge",
                            evidence=(
                                f"Bridge concept '{bridge}' links {left_domain} and {right_domain} "
                                f"within {document.source}."
                            ),
                            reasoning_trace_id=uuid4(),
                            confidence=0.86,
                            metadata={
                                "strategy": "domain_network",
                                "bridge_concept": bridge,
                                "from": left_domain,
                                "to": right_domain,
                            },
                            created_at=datetime.now(UTC),
                        )
                    )

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

    def _tokenize(self, text: str) -> list[str]:
        return re.findall(r"[a-z][a-z0-9_+-]{2,}", text.lower())

    def _extract_domain_concepts(self, tokens: list[str]) -> dict[str, list[str]]:
        domain_scores: dict[str, int] = {}
        for domain, keywords in self._DOMAIN_KEYWORDS.items():
            score = sum(1 for token in tokens if token in keywords)
            if score > 0:
                domain_scores[domain] = score

        if len(domain_scores) < 2:
            return {}

        token_counts = Counter(
            token
            for token in tokens
            if token not in self._STOPWORDS and len(token) >= 4 and not token.isdigit()
        )
        global_concepts = [token for token, _ in token_counts.most_common(10)]

        domain_to_concepts: dict[str, list[str]] = {}
        for domain in domain_scores:
            domain_keywords = self._DOMAIN_KEYWORDS[domain]
            domain_specific = [token for token in token_counts if token in domain_keywords]
            shared_candidates = [token for token in global_concepts if token not in self._STOPWORDS]

            merged: list[str] = []
            for token in domain_specific + shared_candidates:
                if token not in merged:
                    merged.append(token)
                if len(merged) >= 8:
                    break

            if merged:
                domain_to_concepts[domain] = merged

        if len(domain_to_concepts) < 2:
            return {}

        return domain_to_concepts

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
