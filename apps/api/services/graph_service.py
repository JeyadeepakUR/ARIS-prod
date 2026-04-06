"""Graph orchestration service for Sprint 3 pipeline."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
import re
from collections import Counter
from itertools import combinations
from uuid import UUID, uuid4

from sqlalchemy.ext.asyncio import AsyncSession

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


@dataclass(frozen=True)
class GraphBuildArtifacts:
    """In-memory outputs from a full graph build run."""

    graph: KnowledgeGraph
    plan_actions: list[ResearchAction]
    trace: dict[str, object]


class GraphService:
    """Build knowledge graph and draft plans from ingested workspace documents."""

    _DOMAIN_KEYWORDS: dict[str, set[str]] = {
        "machine_learning": {
            "ml",
            "ai",
            "model",
            "neural",
            "learning",
            "classification",
            "prediction",
            "inference",
            "deep",
            "transformer",
            "embedding",
            "training",
            "dataset",
        },
        "cybersecurity": {
            "security",
            "cyber",
            "threat",
            "attack",
            "malware",
            "anomaly",
            "intrusion",
            "encryption",
            "privacy",
            "forensics",
            "audit",
            "trust",
        },
        "blockchain": {
            "blockchain",
            "ledger",
            "smart",
            "contract",
            "consensus",
            "token",
            "immutable",
            "decentralized",
            "proof",
        },
        "healthcare": {
            "cancer",
            "clinical",
            "diagnosis",
            "patient",
            "medical",
            "biometric",
            "depression",
            "therapy",
            "detection",
        },
        "computer_vision": {
            "image",
            "vision",
            "video",
            "spectrogram",
            "detection",
            "segmentation",
            "recognition",
            "captioning",
        },
    }

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

    _LOW_VALUE_TERMS: set[str] = {
        "learning",
        "model",
        "system",
        "method",
        "approach",
        "result",
        "performance",
        "technique",
        "application",
        "using",
        "based",
        "analysis",
        "framework",
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

        materializer = LinkMaterializer(ReasoningEngine(), Evaluator())
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
            source_title = ""
            metadata_source = node.metadata_json.get("source") if isinstance(node.metadata_json, dict) else None
            if isinstance(metadata_source, str):
                source_title = metadata_source

            result = self._classify_node_with_llm_fallback(node.label, source_title)
            node.tier = int(result["tier"])
            cluster = result["cluster_id"]
            node.cluster_id = cluster if isinstance(cluster, str) and cluster else self._derive_cluster(node.label)
            if bool(result["low_value"]):
                node.metadata_json = {**node.metadata_json, "low_value": True}

        for edge in edges:
            source_node = nodes_by_id.get(edge.source_node_id)
            target_node = nodes_by_id.get(edge.target_node_id)
            if source_node is None or target_node is None:
                edge.edge_category = "INTRA_DOMAIN"
                continue

            if abs(source_node.tier - target_node.tier) == 1 and (
                source_node.cluster_id == target_node.cluster_id
            ):
                edge.edge_category = "HIERARCHICAL"
            elif (source_node.cluster_id or "") == (target_node.cluster_id or ""):
                edge.edge_category = "INTRA_DOMAIN"
            else:
                edge.edge_category = "INTER_DOMAIN_BRIDGE"

            if edge.edge_category == "INTER_DOMAIN_BRIDGE" and not edge.bridge_concept:
                edge.bridge_concept = self._synthesize_bridge_concept_with_llm_fallback(source_node, target_node, edge)

        await db.flush()

    def _classify_node_with_llm_fallback(self, label: str, source_title: str) -> dict[str, object]:
        normalized_label = label.strip().lower()
        normalized_source = source_title.strip().lower()

        if normalized_label in self._LOW_VALUE_TERMS:
            return {"tier": 3, "cluster_id": self._derive_cluster(source_title or label), "low_value": True}

        if any(token in normalized_label for token in ["domain", "field", "research area"]):
            return {"tier": 1, "cluster_id": self._derive_cluster(label), "low_value": False}

        if any(token in normalized_label for token in ["detection", "learning", "security", "contracts"]):
            return {"tier": 2, "cluster_id": self._derive_cluster(label), "low_value": False}

        return {
            "tier": 3,
            "cluster_id": self._derive_cluster(f"{label} {normalized_source}"),
            "low_value": False,
        }

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
        evidence_words = [token for token in re.findall(r"[a-z]{4,}", edge.evidence.lower()) if token not in self._STOPWORDS]
        keyword = evidence_words[0] if evidence_words else "integration"
        return f"{source_cluster.lower()} {target_cluster.lower()} {keyword}".strip()

    def _to_core_document(self, document: Document) -> CoreDocument:
        metadata = {k: str(v) for k, v in document.metadata_json.items()}
        content = self._synthesize_content(document, metadata)
        return CoreDocument(
            content=content,
            source=document.filename,
            format=document.file_format,
            metadata=metadata,
            document_id=document.id,
            created_at=datetime.now(UTC),
        )

    def _synthesize_content(self, document: Document, metadata: dict[str, str]) -> str:
        source = metadata.get("source", document.filename)
        preview = metadata.get("content_preview", "")
        preview_compact = re.sub(r"\s+", " ", preview).strip()
        preview_compact = preview_compact[:1500]
        return (
            f"Document {document.filename} format={document.file_format} "
            f"workspace={document.workspace_id} source={source} "
            f"content={preview_compact}"
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
        for domain, keywords in domain_scores.items():
            domain_keywords = self._DOMAIN_KEYWORDS[domain]
            domain_specific = [token for token in token_counts if token in domain_keywords]
            shared_candidates = [token for token in global_concepts if token not in self._STOPWORDS]

            merged = []
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
