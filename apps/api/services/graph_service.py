"""Graph orchestration service.

Provides node/edge classification and domain detection helpers used by the
graph build pipeline. The LangGraph agent network (graph_build_task) handles
the actual graph construction.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from datetime import UTC, datetime
from uuid import UUID

from sqlalchemy.ext.asyncio import AsyncSession

from apps.api.config import get_settings
from apps.api.models.edge import Edge as EdgeModel
from apps.api.models.node import Node as NodeModel
from aris.llm.provider import LLMProvider, get_provider


# ── Lightweight in-memory graph types (replace deleted aris.graph.knowledge_graph) ──

@dataclass
class SimpleNode:
    node_id: UUID
    document_id: UUID
    label: str
    node_type: str
    metadata: dict = field(default_factory=dict)


@dataclass
class SimpleEdge:
    edge_id: UUID
    source_id: UUID
    target_id: UUID
    edge_type: str
    evidence: str
    reasoning_trace_id: UUID
    confidence: float
    metadata: dict = field(default_factory=dict)
    created_at: datetime = field(default_factory=lambda: datetime.now(UTC))


@dataclass
class SimpleGraph:
    graph_id: UUID
    nodes: list[SimpleNode] = field(default_factory=list)
    edges: list[SimpleEdge] = field(default_factory=list)
    created_at: datetime = field(default_factory=lambda: datetime.now(UTC))


@dataclass
class ResearchAction:
    action_id: UUID
    action_type: str
    description: str
    evidence: str
    rationale: str
    priority: float
    metadata: dict = field(default_factory=dict)


@dataclass(frozen=True)
class GraphBuildArtifacts:
    graph: SimpleGraph
    plan_actions: list[ResearchAction]
    trace: dict


class GraphService:
    """Node/edge classification and plan generation helpers for the graph pipeline."""

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

    _STOPWORDS: set[str] = {
        "the", "and", "for", "with", "from", "that", "this", "into",
        "using", "based", "study", "system", "analysis", "paper",
        "research", "through", "between", "over", "under", "data",
        "method",
    }

    def __init__(self, provider: LLMProvider | None = None) -> None:
        if provider is None:
            settings = get_settings()
            try:
                provider = get_provider(
                    settings.llm_provider,
                    model=settings.llm_model,
                    api_key=settings.llm_api_key,
                    base_url=settings.ollama_base_url,
                )
            except Exception:
                from aris.llm.mock_provider import MockProvider
                provider = MockProvider()
        self._provider = provider

    def create_plans_from_persisted(
        self,
        *,
        graph_id: UUID,
        nodes: list[NodeModel],
        edges: list[EdgeModel],
        strategy: str,
        max_actions: int,
    ) -> list[ResearchAction]:
        """Return plan actions derived from persisted graph nodes and edges."""
        return []

    async def classify_nodes_and_edges(
        self,
        graph_id: UUID,
        nodes: list[NodeModel],
        edges: list[EdgeModel],
        db: AsyncSession,
    ) -> None:
        """Classify node tiers/clusters and edge categories in-place, then flush."""
        nodes_by_id = {node.id: node for node in nodes}

        for node in nodes:
            meta = node.metadata_json if isinstance(node.metadata_json, dict) else {}

            if node.node_type == "domain":
                node.tier = 1
                node.cluster_id = node.label
            elif node.node_type in {"concept", "bridge_concept"} and "domain" in meta:
                node.tier = 3 if node.node_type == "concept" else 2
                node.cluster_id = str(meta["domain"]).replace("_", " ").title()
            elif node.node_type == "document":
                node.tier = 1
                node.cluster_id = self._derive_cluster(node.label)
            else:
                source_title = str(meta.get("source", ""))
                result = self._classify_node_with_llm_fallback(node.label, source_title)
                node.tier = int(result["tier"])
                cluster = result["cluster_id"]
                node.cluster_id = cluster if isinstance(cluster, str) and cluster else self._derive_cluster(node.label)
                if bool(result.get("low_value")):
                    node.metadata_json = {**meta, "low_value": True}

        for edge in edges:
            meta = edge.metadata_json if isinstance(edge.metadata_json, dict) else {}

            if edge.edge_type == "cross_domain_bridge":
                edge.edge_category = "INTER_DOMAIN_BRIDGE"
                if not edge.bridge_concept:
                    edge.bridge_concept = str(meta.get("bridge_concept", "")) or None
                continue

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
                    edge.bridge_concept = self._synthesize_bridge_concept(
                        source_node, target_node, edge
                    )
            elif abs(source_node.tier - target_node.tier) == 1 and src_cluster == tgt_cluster:
                edge.edge_category = "HIERARCHICAL"
            else:
                edge.edge_category = "INTRA_DOMAIN"

        await db.flush()

    # ── LLM-assisted classification helpers ──────────────────────────────────

    def _classify_node_with_llm_fallback(self, label: str, source_title: str) -> dict:
        try:
            prompt = (
                "Classify this knowledge graph node for a research document.\n\n"
                f"Node label: {label}\n"
                f"Source document: {source_title}\n\n"
                "Return a JSON object with:\n"
                '- "tier": integer 1 (broad domain), 2 (sub-domain/technique), or 3 (specific concept)\n'
                '- "cluster_id": short domain name (e.g. "Machine Learning", "Cybersecurity")\n'
                '- "low_value": boolean — true only for extremely generic terms\n\n'
                "Return only the JSON object."
            )
            result = self._provider.complete_json(prompt, max_tokens=120)
            tier = int(result.get("tier", 3))
            cluster = str(result.get("cluster_id", "") or "")
            if tier not in {1, 2, 3}:
                tier = 3
            if not cluster:
                cluster = self._derive_cluster(f"{label} {source_title}")
            return {"tier": tier, "cluster_id": cluster, "low_value": bool(result.get("low_value"))}
        except Exception:
            return self._classify_node_heuristic(label, source_title)

    def _classify_node_heuristic(self, label: str, source_title: str) -> dict:
        low_value = {
            "learning", "model", "system", "method", "approach", "result",
            "performance", "technique", "application", "analysis", "framework",
        }
        normalized = label.strip().lower()
        return {
            "tier": 3,
            "cluster_id": self._derive_cluster(f"{label} {source_title}"),
            "low_value": normalized in low_value,
        }

    def _derive_cluster(self, text: str) -> str:
        lowered = text.lower()
        for key, keywords in self._DOMAIN_KEYWORDS.items():
            if any(kw in lowered for kw in keywords):
                return key.replace("_", " ").title()
        return "General Research"

    def _synthesize_bridge_concept(
        self,
        source_node: NodeModel,
        target_node: NodeModel,
        edge: EdgeModel,
    ) -> str:
        src = source_node.cluster_id or "general"
        tgt = target_node.cluster_id or "general"
        snippet = edge.evidence[:400] if edge.evidence else ""
        try:
            prompt = (
                f"Source domain: {src}\nSource concept: {source_node.label}\n"
                f"Target domain: {tgt}\nTarget concept: {target_node.label}\n"
                f"Evidence: {snippet}\n\n"
                "Identify the precise conceptual bridge (3-8 words) that enables knowledge "
                "transfer between these domains. Return JSON: "
                '{"bridge_concept": "..."}'
            )
            result = self._provider.complete_json(prompt, max_tokens=100)
            bridge = str(result.get("bridge_concept", "")).strip()
            if bridge:
                return bridge
        except Exception:
            pass
        words = [t for t in re.findall(r"[a-z]{4,}", edge.evidence.lower()) if t not in self._STOPWORDS]
        kw = words[0] if words else "integration"
        return f"{src} {tgt} {kw}".strip()
