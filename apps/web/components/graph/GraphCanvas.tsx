"use client";

import { useMemo, useState } from "react";
import ReactFlow, {
  Background,
  Controls,
  MiniMap,
  MarkerType,
  Node,
  Edge,
  NodeTypes,
} from "reactflow";

import type { GraphEdge, GraphNode } from "../../lib/api/graphs";
import type { ARISEdge, ARISNode, ViewMode } from "../../lib/graph/layout";
import { colorForDomain, computeGraphLayout } from "../../lib/graph/layout";
import { GraphLegend } from "./GraphLegend";
import { useGraphStore } from "../../lib/stores/graphStore";
import { BridgeMarkerNode } from "./BridgeMarkerNode";
import { BridgeEdge } from "./BridgeEdge";
import { ConceptNode } from "./ConceptNode";
import { DocumentNode } from "./DocumentNode";
import { DomainNode } from "./DomainNode";
import { EdgeInspector } from "./EdgeInspector";
import { HypothesisPanel } from "./HypothesisPanel";
import { IntraDomainEdge } from "./IntraDomainEdge";
import { NodeCard } from "./NodeCard";
import { NodeInspector } from "./NodeInspector";
import { SubDomainNode } from "./SubDomainNode";

const nodeTypes: NodeTypes = {
  nodeCard: ({ data }) => (
    <NodeCard label={String(data.label)} nodeType={String(data.nodeType)} />
  ),
  domain: DomainNode,
  document: DocumentNode,
  subdomain: SubDomainNode,
  concept: ConceptNode,
  bridge_marker: BridgeMarkerNode,
};

const edgeTypes = {
  intra_domain: IntraDomainEdge,
  bridge: BridgeEdge,
  has_concept: IntraDomainEdge,
  extracted_from: IntraDomainEdge,
};

type GraphCanvasProps = {
  graphId: string;
  nodes: GraphNode[];
  edges: GraphEdge[];
  confidenceThreshold: number;
  bridgeFocus: boolean;
  viewMode: ViewMode;
  onSelectEdge: (edgeId: string) => void;
};

export function GraphCanvas({
  graphId,
  nodes,
  edges,
  confidenceThreshold,
  bridgeFocus,
  viewMode,
  onSelectEdge,
}: GraphCanvasProps) {
  const activeBridgeEdgeId = useGraphStore((state) => state.activeBridgeEdgeId);
  const setActiveBridge = useGraphStore((state) => state.setActiveBridge);
  const getEdgeOpacity = useGraphStore((state) => state.getEdgeOpacity);
  const getNodeOpacity = useGraphStore((state) => state.getNodeOpacity);

  const [inspectedNodeId, setInspectedNodeId] = useState<string | null>(null);
  const [inspectedEdgeId, setInspectedEdgeId] = useState<string | null>(null);
  const [hoveredNodeId, setHoveredNodeId] = useState<string | null>(null);

  // Concept counts per domain hub (used for badges on DomainNode).
  const conceptCountByDomain = useMemo(() => {
    const counts = new Map<string, number>();
    for (const e of edges) {
      if (e.edge_type === "has_concept") {
        counts.set(e.source_node_id, (counts.get(e.source_node_id) ?? 0) + 1);
      }
    }
    return counts;
  }, [edges]);

  // We keep all edges in the layout (just dimming weak ones) so the user can
  // see "there's more here" instead of edges silently disappearing.
  const filteredEdges = useMemo(() => {
    if (viewMode === "bridges-only") {
      return edges.filter(
        (edge) =>
          edge.edge_type === "cross_domain_bridge" ||
          edge.edge_type === "has_concept" ||
          edge.edge_type === "extracted_from",
      );
    }
    if (bridgeFocus) {
      return edges.filter((edge) => {
        if (edge.edge_type === "has_concept" || edge.edge_type === "extracted_from") return true;
        return (
          edge.edge_type === "cross_domain_bridge" || edge.edge_type === "belongs_to_domain"
        );
      });
    }
    return edges;
  }, [edges, bridgeFocus, viewMode]);

  const visibleNodes = useMemo(() => {
    if (filteredEdges.length === 0) return nodes;

    const endpointIds = new Set<string>();
    filteredEdges.forEach((edge) => {
      endpointIds.add(edge.source_node_id);
      endpointIds.add(edge.target_node_id);
    });

    return nodes.filter(
      (node) =>
        endpointIds.has(node.id) ||
        node.node_type === "domain" ||
        node.node_type === "document" ||
        (node.document_id ? endpointIds.has(node.document_id) : false),
    );
  }, [nodes, filteredEdges]);

  const arisNodes = useMemo<ARISNode[]>(
    () =>
      visibleNodes.map((node) => ({
        id: node.id,
        position: { x: 0, y: 0 },
        type: node.node_type,
        data: {
          label: node.label,
          tier: node.tier,
          cluster_id: node.cluster_id ?? "General Research",
          document_id: node.document_id ?? undefined,
          low_value: Boolean((node.metadata?.low_value as boolean | undefined) ?? false),
          node_type: node.node_type,
        },
      })),
    [visibleNodes],
  );

  const arisEdges = useMemo<ARISEdge[]>(
    () =>
      filteredEdges.map((edge) => ({
        id: edge.id,
        source: edge.source_node_id,
        target: edge.target_node_id,
        data: {
          edge_category: edge.edge_category,
          edge_type: edge.edge_type,
          confidence: edge.confidence,
          bridge_concept: edge.bridge_concept ?? undefined,
          evidence:
            typeof edge.evidence?.text === "string"
              ? edge.evidence.text
              : JSON.stringify(edge.evidence),
        },
      })),
    [filteredEdges],
  );

  const positionedNodes = useMemo(
    () => computeGraphLayout(arisNodes, arisEdges, viewMode),
    [arisNodes, arisEdges, viewMode],
  );

  // ── Hover affordances ──────────────────────────────────────────────────
  // When the user hovers a node, we compute the set of "related" node IDs:
  //   - the node itself
  //   - everything connected by any direct edge
  //   - if the hovered node is a domain hub, all its concepts
  //   - if the hovered node is a document hub, all concepts extracted from it
  const relatedToHover = useMemo<Set<string> | null>(() => {
    if (!hoveredNodeId) return null;
    const set = new Set<string>([hoveredNodeId]);
    for (const e of edges) {
      if (e.source_node_id === hoveredNodeId) set.add(e.target_node_id);
      if (e.target_node_id === hoveredNodeId) set.add(e.source_node_id);
    }
    return set;
  }, [hoveredNodeId, edges]);

  const dimEdge = (edge: ARISEdge): boolean => {
    if (!relatedToHover) return false;
    return !(relatedToHover.has(edge.source) && relatedToHover.has(edge.target));
  };
  const dimNode = (id: string): boolean => {
    if (!relatedToHover) return false;
    return !relatedToHover.has(id);
  };

  const flowNodes = useMemo<Node[]>(() => {
    return positionedNodes.map((node) => {
      const nodeType = node.data.node_type === "domain"
        ? "domain"
        : node.data.node_type === "document"
        ? "document"
        : node.data.node_type === "bridge_concept"
        ? "concept"
        : "concept";

      const extra =
        node.data.node_type === "domain"
          ? { concept_count: conceptCountByDomain.get(node.id) ?? 0 }
          : {};

      const baseOpacity = getNodeOpacity(node);
      const opacity = dimNode(node.id) ? Math.min(baseOpacity, 0.25) : baseOpacity;

      return {
        id: node.id,
        type: nodeType,
        position: node.position,
        data: { ...node.data, ...extra },
        style: { opacity, transition: "opacity 0.18s ease" },
      } as Node;
    });
    // dimNode is computed from hoveredNodeId/edges; depend on those.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [positionedNodes, conceptCountByDomain, getNodeOpacity, hoveredNodeId, edges]);

  const flowEdges = useMemo<Edge[]>(() => {
    const nodeById = new Map(positionedNodes.map((node) => [node.id, node]));
    return arisEdges.map((edge) => {
      const sourceNode = nodeById.get(edge.source);
      const targetNode = nodeById.get(edge.target);
      const isBridge = edge.data?.edge_category === "INTER_DOMAIN_BRIDGE";
      const isHasConcept = edge.data?.edge_type === "has_concept";
      const isExtractedFrom = edge.data?.edge_type === "extracted_from";

      let edgeType = "intra_domain";
      if (isBridge) edgeType = "bridge";

      const sourceCluster = sourceNode?.data.cluster_id ?? "General Research";
      const sourceColor = colorForDomain(sourceCluster);

      let stroke = "rgba(255,255,255,0.10)";
      let strokeWidth = 1;
      let strokeDasharray: string | undefined;
      if (isHasConcept) {
        stroke = `${sourceColor}88`;
        strokeWidth = 1.4;
      } else if (isExtractedFrom) {
        stroke = "rgba(148,163,184,0.35)";
        strokeWidth = 0.9;
        strokeDasharray = "3 3";
      }

      const confidence = edge.data?.confidence ?? 1;
      const isStructural = isHasConcept || isExtractedFrom;
      // Confidence-based dimming, but never make weak edges invisible — leave
      // a faint trace so the user knows there's more to explore.
      const confidenceMultiplier = isStructural
        ? 1
        : confidence >= confidenceThreshold
        ? 1
        : 0.18;
      const baseOpacity = getEdgeOpacity(edge) * confidenceMultiplier;
      const opacity = dimEdge(edge) ? Math.min(baseOpacity, 0.08) : baseOpacity;

      return {
        id: edge.id,
        source: edge.source,
        target: edge.target,
        type: edgeType,
        markerEnd: { type: MarkerType.ArrowClosed, color: stroke },
        data: {
          ...edge.data,
          source_cluster_id: sourceNode?.data.cluster_id,
          target_cluster_id: targetNode?.data.cluster_id,
        },
        style: {
          opacity,
          transition: "opacity 0.18s ease",
          ...(isStructural ? { stroke, strokeWidth, strokeDasharray } : {}),
        },
      } as Edge;
    });
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [arisEdges, getEdgeOpacity, positionedNodes, hoveredNodeId, edges, confidenceThreshold]);

  const onEdgeClick = (_: React.MouseEvent, edge: Edge) => {
    onSelectEdge(edge.id);
    // Open the edge inspector for any meaningful relationship; structural
    // hierarchy edges (has_concept / extracted_from) carry no extra evidence.
    const isInspectable =
      edge.data?.edge_type === "cross_domain_bridge" ||
      edge.data?.edge_category === "INTER_DOMAIN_BRIDGE";
    if (isInspectable) {
      setInspectedEdgeId(edge.id);
      setInspectedNodeId(null);
      // Suppress the legacy hypothesis side-panel; the EdgeInspector replaces it.
      setActiveBridge(null);
    } else {
      setActiveBridge(null);
    }
  };

  const onNodeClick = (_: React.MouseEvent, n: Node) => {
    setInspectedNodeId(n.id);
    setInspectedEdgeId(null);
    // Clear any lingering bridge selection so the hypothesis panel doesn't
    // stack behind the node inspector.
    setActiveBridge(null);
  };

  const onPaneClick = () => {
    setInspectedNodeId(null);
    setInspectedEdgeId(null);
    setActiveBridge(null);
  };

  const inspectedNode = inspectedNodeId
    ? nodes.find((n) => n.id === inspectedNodeId) ?? null
    : null;
  const inspectedEdge = inspectedEdgeId
    ? edges.find((e) => e.id === inspectedEdgeId) ?? null
    : null;

  const onNodeMouseEnter = (_: React.MouseEvent, n: Node) => setHoveredNodeId(n.id);
  const onNodeMouseLeave = () => setHoveredNodeId(null);

  const activeDomains = useMemo(() => {
    const set = new Set<string>();
    for (const n of nodes) {
      if (n.cluster_id) set.add(n.cluster_id);
    }
    return Array.from(set);
  }, [nodes]);

  return (
    <div style={{ width: "100%", height: "100%", background: "#0d0e14", position: "relative" }}>
      <ReactFlow
        nodes={flowNodes}
        edges={flowEdges}
        fitView
        fitViewOptions={{ padding: 0.18 }}
        minZoom={0.18}
        maxZoom={2}
        nodeTypes={nodeTypes}
        edgeTypes={edgeTypes}
        onEdgeClick={onEdgeClick}
        onNodeClick={onNodeClick}
        onNodeMouseEnter={onNodeMouseEnter}
        onNodeMouseLeave={onNodeMouseLeave}
        onPaneClick={onPaneClick}
        proOptions={{ hideAttribution: true }}
      >
        <Background color="rgba(255,255,255,0.04)" gap={24} size={1} />
        <MiniMap
          nodeColor={(n) => {
            const t = String(n.data?.nodeType ?? n.type ?? "");
            if (t === "domain") return colorForDomain(String(n.data?.cluster_id ?? ""));
            if (t === "document") return "#94a3b8";
            return colorForDomain(String(n.data?.cluster_id ?? ""));
          }}
          nodeStrokeWidth={2}
          maskColor="rgba(10,11,20,0.65)"
          position="bottom-right"
          zoomable
          pannable
        />
        <Controls />
      </ReactFlow>

      {inspectedEdge ? (
        <EdgeInspector
          edge={inspectedEdge}
          graphId={graphId}
          allNodes={nodes}
          onClose={() => setInspectedEdgeId(null)}
        />
      ) : (
        <NodeInspector
          node={inspectedNode}
          graphId={graphId}
          allNodes={nodes}
          allEdges={edges}
          onClose={() => setInspectedNodeId(null)}
          onFocusNode={(id) => {
            setInspectedNodeId(id);
            setInspectedEdgeId(null);
          }}
        />
      )}

      {activeBridgeEdgeId && !inspectedEdge && !inspectedNode ? (
        <HypothesisPanel edges={edges} nodes={nodes} />
      ) : null}

      <GraphLegend activeDomains={activeDomains} />
    </div>
  );
}
