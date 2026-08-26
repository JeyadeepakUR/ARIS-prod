import dagre from "dagre";

export type ARISNode = {
  id: string;
  position: { x: number; y: number };
  type: string;
  data: {
    label: string;
    tier: number;
    cluster_id: string;
    document_id?: string;
    low_value: boolean;
    node_type: string;
  };
};

export type ARISEdge = {
  id: string;
  source: string;
  target: string;
  data?: {
    edge_category: string;
    edge_type?: string;
    confidence: number;
    bridge_concept?: string;
    evidence?: string;
  };
};

// Approximate footprint per node-type so dagre can lay them out cleanly.
const NODE_SIZE: Record<string, { width: number; height: number }> = {
  domain: { width: 220, height: 70 },
  document: { width: 220, height: 70 },
  bridge_concept: { width: 200, height: 60 },
  concept: { width: 200, height: 50 },
};

export type ViewMode = "by-domain" | "by-paper" | "bridges-only";

/**
 * Hierarchical, domain-aware layout using dagre.
 *
 * View modes:
 *   - "by-domain"    Documents → Domain hubs → Concepts (default)
 *   - "by-paper"     Domain hubs → Documents → Concepts; concepts cluster
 *                    under the paper that produced them and inter-paper
 *                    relationships read as cross-paper arcs
 *   - "bridges-only" Show only nodes touching a cross-domain bridge edge,
 *                    laid out as a focused subgraph
 */
export function computeGraphLayout(
  nodes: ARISNode[],
  edges: ARISEdge[],
  viewMode: ViewMode = "by-domain",
): ARISNode[] {
  if (nodes.length === 0) return [];

  // Bridges-only mode: prune to nodes touched by a bridge edge plus their
  // immediate neighbours so the user sees an isolated cross-domain story.
  let workingNodes = nodes;
  let workingEdges = edges;
  if (viewMode === "bridges-only") {
    const bridgeEdges = edges.filter(
      (e) => e.data?.edge_type === "cross_domain_bridge" || e.data?.edge_category === "INTER_DOMAIN_BRIDGE",
    );
    const keep = new Set<string>();
    for (const be of bridgeEdges) {
      keep.add(be.source);
      keep.add(be.target);
    }
    if (keep.size === 0) {
      // Graceful fall-back: nothing to show in bridges-only mode.
      return [];
    }
    // Pull in the domain hubs the kept nodes belong to so users keep their
    // context anchors.
    const conceptDomains = new Set<string>();
    for (const n of nodes) {
      if (keep.has(n.id) && n.data.node_type === "concept") {
        conceptDomains.add(n.data.cluster_id);
      }
    }
    for (const n of nodes) {
      if (n.data.node_type === "domain" && conceptDomains.has(n.data.cluster_id)) {
        keep.add(n.id);
      }
    }
    workingNodes = nodes.filter((n) => keep.has(n.id));
    workingEdges = edges.filter((e) => keep.has(e.source) && keep.has(e.target));
  }

  const g = new dagre.graphlib.Graph({ multigraph: true });
  g.setGraph({
    rankdir: "LR",
    // Tighter spacing in bridges-only mode since there are fewer nodes.
    ranksep: viewMode === "bridges-only" ? 200 : 140,
    nodesep: viewMode === "by-paper" ? 36 : 28,
    edgesep: 20,
    marginx: 60,
    marginy: 40,
  });
  g.setDefaultEdgeLabel(() => ({}));

  // Add nodes with node-type-specific sizes
  for (const node of workingNodes) {
    const size = NODE_SIZE[node.data.node_type] ?? NODE_SIZE.concept;
    g.setNode(node.id, { width: size.width, height: size.height });
  }

  const documentIds = workingNodes.filter((n) => n.data.node_type === "document").map((n) => n.id);
  const domainIds = workingNodes.filter((n) => n.data.node_type === "domain").map((n) => n.id);

  if (viewMode === "by-paper") {
    // Pin domains BEFORE documents (so visual flow becomes domain → paper → concept).
    for (const domId of domainIds) {
      for (const docId of documentIds) {
        g.setEdge(domId, docId, { weight: 0.01, minlen: 1 }, `__layer-${domId}-${docId}`);
      }
    }
  } else {
    // Default ordering: documents → domains → concepts
    for (const docId of documentIds) {
      for (const domId of domainIds) {
        g.setEdge(docId, domId, { weight: 0.01, minlen: 1 }, `__layer-${docId}-${domId}`);
      }
    }
  }

  // Add real edges so dagre keeps connected nodes nearby.
  for (const edge of workingEdges) {
    if (!g.hasNode(edge.source) || !g.hasNode(edge.target)) continue;
    // Skip extracted_from — concept→document creates a cycle with the
    // document→domain layer edges above, causing dagre to produce NaN positions.
    if (edge.data?.edge_type === "extracted_from") continue;
    const isBridge =
      edge.data?.edge_type === "cross_domain_bridge" ||
      edge.data?.edge_category === "INTER_DOMAIN_BRIDGE";
    g.setEdge(
      edge.source,
      edge.target,
      { weight: isBridge ? 3 : 1 },
      edge.id,
    );
  }

  try {
    dagre.layout(g);
  } catch {
    return gridFallback(workingNodes);
  }

  const positioned = workingNodes.map((node) => {
    const layout = g.node(node.id);
    if (!layout) return { ...node, position: { x: 0, y: 0 } };
    const size = NODE_SIZE[node.data.node_type] ?? NODE_SIZE.concept;
    return {
      ...node,
      position: {
        x: layout.x - size.width / 2,
        y: layout.y - size.height / 2,
      },
    };
  });

  return positioned;
}

function gridFallback(nodes: ARISNode[]): ARISNode[] {
  const cols = Math.ceil(Math.sqrt(nodes.length));
  return nodes.map((node, idx) => ({
    ...node,
    position: {
      x: (idx % cols) * 240,
      y: Math.floor(idx / cols) * 100,
    },
  }));
}

// ── Domain colour palette (used by ConceptNode + DomainNode + edges) ──────
export const DOMAIN_COLORS: Record<string, string> = {
  "Machine Learning": "#6366f1",
  "Deep Learning": "#8b5cf6",
  "Natural Language Processing": "#ec4899",
  "Computer Vision": "#06b6d4",
  "Reinforcement Learning": "#f59e0b",
  "Cybersecurity": "#ef4444",
  "Blockchain": "#10b981",
  "Healthcare": "#22d3ee",
  "Bioinformatics": "#84cc16",
  "Robotics": "#f97316",
  "Information Retrieval": "#a855f7",
  "Data Engineering": "#14b8a6",
  "Distributed Systems": "#0ea5e9",
  "Software Engineering": "#64748b",
  "Statistics": "#fb923c",
  "Optimization": "#facc15",
  "Quantum Computing": "#d946ef",
  "General Research": "#94a3b8",
  "Source Documents": "#94a3b8",
};

export function colorForDomain(domain: string | null | undefined): string {
  if (!domain) return DOMAIN_COLORS["General Research"];
  return DOMAIN_COLORS[domain] ?? DOMAIN_COLORS["General Research"];
}
