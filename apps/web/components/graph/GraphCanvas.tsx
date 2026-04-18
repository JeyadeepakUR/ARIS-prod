"use client";

import { useMemo } from "react";
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
import type { ARISEdge, ARISNode } from "../../lib/graph/layout";
import { computeGraphLayout } from "../../lib/graph/layout";
import { useGraphStore } from "../../lib/stores/graphStore";
import { BridgeMarkerNode } from "./BridgeMarkerNode";
import { BridgeEdge } from "./BridgeEdge";
import { ConceptNode } from "./ConceptNode";
import { DomainNode } from "./DomainNode";
import { HypothesisPanel } from "./HypothesisPanel";
import { IntraDomainEdge } from "./IntraDomainEdge";
import { NodeCard } from "./NodeCard";
import { SubDomainNode } from "./SubDomainNode";

const nodeTypes: NodeTypes = {
  nodeCard: ({ data }) => <NodeCard label={String(data.label)} nodeType={String(data.nodeType)} />,
  domain: DomainNode,
  subdomain: SubDomainNode,
  concept: ConceptNode,
  bridge_marker: BridgeMarkerNode,
};

const edgeTypes = {
  intra_domain: IntraDomainEdge,
  bridge: BridgeEdge,
};

type GraphCanvasProps = {
  nodes: GraphNode[];
  edges: GraphEdge[];
  confidenceThreshold: number;
  bridgeFocus: boolean;
  onSelectEdge: (edgeId: string) => void;
};

function buildFlowNodes(nodes: GraphNode[], edges: GraphEdge[]): Node[] {
  const nodeById = new Map(nodes.map((node) => [node.id, node]));

  const domainToConcepts = new Map<string, string[]>();
  const domainToDocuments = new Map<string, string[]>();
  const bridgeToDomains = new Map<string, string[]>();

  function add(map: Map<string, string[]>, key: string, value: string) {
    const current = map.get(key) ?? [];
    if (!current.includes(value)) {
      current.push(value);
      map.set(key, current);
    }
  }

  for (const edge of edges) {
    const source = nodeById.get(edge.source_node_id);
    const target = nodeById.get(edge.target_node_id);
    if (!source || !target) {
      continue;
    }

    if (edge.edge_type === "has_concept" && source.node_type === "domain") {
      add(domainToConcepts, source.id, target.id);
      continue;
    }

    if (edge.edge_type === "belongs_to_domain") {
      if (source.node_type === "document" && target.node_type === "domain") {
        add(domainToDocuments, target.id, source.id);
      } else if (source.node_type === "domain" && target.node_type === "document") {
        add(domainToDocuments, source.id, target.id);
      }
      continue;
    }

    if (edge.edge_type === "cross_domain_bridge") {
      if (source.node_type === "domain" && target.node_type === "bridge_concept") {
        add(bridgeToDomains, target.id, source.id);
      } else if (source.node_type === "bridge_concept" && target.node_type === "domain") {
        add(bridgeToDomains, source.id, target.id);
      }
    }
  }

  const priority: Record<string, number> = {
    document: 0,
    domain: 1,
    bridge_concept: 2,
    concept: 3,
  };

  const sorted = [...nodes].sort((a, b) => {
    const typeDiff = (priority[a.node_type] ?? 99) - (priority[b.node_type] ?? 99);
    if (typeDiff !== 0) {
      return typeDiff;
    }
    return a.label.localeCompare(b.label);
  });

  const yByDomain = new Map<string, number>();
  const positionByNodeId = new Map<string, { x: number; y: number }>();

  const domainNodes = sorted.filter((node) => node.node_type === "domain");
  const documentNodes = sorted.filter((node) => node.node_type === "document");
  const bridgeNodes = sorted.filter((node) => node.node_type === "bridge_concept");

  let yCursor = 80;
  for (const domain of domainNodes) {
    const conceptsCount = (domainToConcepts.get(domain.id) ?? []).length;
    const docsCount = (domainToDocuments.get(domain.id) ?? []).length;
    const blockHeight = Math.max(260, Math.max(conceptsCount * 84, docsCount * 96));
    const yCenter = yCursor + blockHeight / 2;

    yByDomain.set(domain.id, yCenter);
    positionByNodeId.set(domain.id, { x: 520, y: yCenter });

    const domainDocs = (domainToDocuments.get(domain.id) ?? [])
      .map((id) => nodeById.get(id))
      .filter((node): node is GraphNode => Boolean(node))
      .sort((a, b) => a.label.localeCompare(b.label));
    domainDocs.forEach((doc, index) => {
      positionByNodeId.set(doc.id, { x: 120, y: yCenter - (domainDocs.length - 1) * 48 + index * 96 });
    });

    const domainConcepts = (domainToConcepts.get(domain.id) ?? [])
      .map((id) => nodeById.get(id))
      .filter((node): node is GraphNode => Boolean(node))
      .filter((node) => node.node_type === "concept")
      .sort((a, b) => a.label.localeCompare(b.label));
    domainConcepts.forEach((concept, index) => {
      positionByNodeId.set(concept.id, {
        x: 980 + (index % 2) * 290,
        y: yCursor + Math.floor(index / 2) * 84,
      });
    });

    yCursor += blockHeight + 90;
  }

  const bridgeYSlots = new Map<number, number>();
  for (const bridge of bridgeNodes) {
    const relatedDomains = (bridgeToDomains.get(bridge.id) ?? [])
      .map((id) => yByDomain.get(id))
      .filter((value): value is number => typeof value === "number");
    const baseY = relatedDomains.length > 0
      ? relatedDomains.reduce((sum, value) => sum + value, 0) / relatedDomains.length
      : 120;
    const slot = Math.round(baseY / 70);
    const stack = bridgeYSlots.get(slot) ?? 0;
    bridgeYSlots.set(slot, stack + 1);
    positionByNodeId.set(bridge.id, {
      x: 740,
      y: slot * 70 + stack * 22,
    });
  }

  let fallbackRow = 0;

  return sorted.map((node) => {
    const fixed = positionByNodeId.get(node.id);
    let x = fixed?.x;
    let y = fixed?.y;

    if (typeof x !== "number" || typeof y !== "number") {
      x = node.node_type === "document" ? 120 : node.node_type === "domain" ? 520 : 1260;
      y = 80 + fallbackRow * 86;
      fallbackRow += 1;
    }

    return {
      id: node.id,
      type: "nodeCard",
      position: { x, y },
      data: {
        label: node.label,
        nodeType: node.node_type,
      },
    };
  });
}

function buildFlowEdges(edges: GraphEdge[], nodes: GraphNode[], confidenceThreshold: number): Edge[] {
  const nodeIdSet = new Set(nodes.map((node) => node.id));
  const documentToNode = new Map(
    nodes
      .filter((node) => Boolean(node.document_id))
      .map((node) => [String(node.document_id), node.id]),
  );

  function resolveNodeId(candidate: string): string {
    if (nodeIdSet.has(candidate)) {
      return candidate;
    }
    return documentToNode.get(candidate) ?? candidate;
  }

  return edges
    .filter((edge) => edge.confidence >= confidenceThreshold)
    .map((edge) => {
      const isBridge = edge.edge_type === "cross_domain_bridge";
      const isConcept = edge.edge_type === "has_concept";
      const stroke = isBridge ? "#b45309" : "#8f4f2b";
      return {
        id: edge.id,
        source: resolveNodeId(edge.source_node_id),
        target: resolveNodeId(edge.target_node_id),
        label: isBridge ? `${edge.edge_type} (${(edge.confidence * 100).toFixed(0)}%)` : undefined,
        labelStyle: { fontSize: 10, fill: "#92400e", fontWeight: 700 },
        markerEnd: { type: MarkerType.ArrowClosed, color: stroke },
        style: {
          strokeWidth: isBridge ? 2.8 : isConcept ? 1.1 : 1.8,
          stroke,
          opacity: isBridge ? 0.98 : isConcept ? 0.38 : 0.62,
        },
        type: "smoothstep",
      };
    });
}

function minimapNodeColor(node: Node): string {
  const nodeType = String(node.data?.nodeType ?? node.type ?? "");
  if (nodeType === "domain") return "#6366f1";
  if (nodeType === "bridge_concept") return "#f97316";
  if (nodeType === "document") return "#818cf8";
  return "#555870";
}

export function GraphCanvas({ nodes, edges, confidenceThreshold, bridgeFocus, onSelectEdge }: GraphCanvasProps) {
  const activeBridgeEdgeId = useGraphStore((state) => state.activeBridgeEdgeId);
  const setActiveBridge = useGraphStore((state) => state.setActiveBridge);
  const getEdgeOpacity = useGraphStore((state) => state.getEdgeOpacity);
  const getNodeOpacity = useGraphStore((state) => state.getNodeOpacity);

  const filteredEdges = useMemo(
    () =>
      edges.filter((edge) => {
        const passesConfidence = edge.confidence >= confidenceThreshold;
        if (!passesConfidence) {
          return false;
        }
        if (!bridgeFocus) {
          return true;
        }
        return edge.edge_type === "cross_domain_bridge" || edge.edge_type === "belongs_to_domain";
      }),
    [edges, confidenceThreshold, bridgeFocus],
  );

  const visibleNodes = useMemo(() => {
    if (filteredEdges.length === 0) {
      return nodes;
    }

    const endpointIds = new Set<string>();
    filteredEdges.forEach((edge) => {
      endpointIds.add(edge.source_node_id);
      endpointIds.add(edge.target_node_id);
    });

    return nodes.filter((node) => endpointIds.has(node.id) || (node.document_id ? endpointIds.has(node.document_id) : false));
  }, [nodes, filteredEdges]);

  const arisNodes = useMemo<ARISNode[]>(
    () =>
      visibleNodes.map((node) => ({
        id: node.id,
        position: { x: 0, y: 0 },
        type: "concept",
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
          confidence: edge.confidence,
          bridge_concept: edge.bridge_concept ?? undefined,
          evidence: typeof edge.evidence?.text === "string" ? edge.evidence.text : JSON.stringify(edge.evidence),
        },
      })),
    [filteredEdges],
  );

  const positionedNodes = useMemo(
    () => computeGraphLayout(arisNodes, arisEdges),
    [arisNodes, arisEdges],
  );

  const flowNodes = useMemo<Node[]>(() => {
    const mapped = positionedNodes.map((node) => {
      const nodeType = node.data.tier === 1 ? "domain" : node.data.tier === 2 ? "subdomain" : "concept";
      return {
        id: node.id,
        type: nodeType,
        position: node.position,
        data: node.data,
        style: {
          opacity: getNodeOpacity(node),
        },
      } as Node;
    });

    const positionByNode = new Map(mapped.map((node) => [node.id, node.position]));
    const bridgeMarkers: Node[] = arisEdges
      .filter((edge) => edge.data?.edge_category === "INTER_DOMAIN_BRIDGE")
      .map((edge) => {
        const sourcePos = positionByNode.get(edge.source);
        const targetPos = positionByNode.get(edge.target);
        const x = sourcePos && targetPos ? (sourcePos.x + targetPos.x) / 2 : 0;
        const y = sourcePos && targetPos ? (sourcePos.y + targetPos.y) / 2 : 0;
        return {
          id: `bridge-marker-${edge.id}`,
          type: "bridge_marker",
          position: { x, y },
          data: {
            edge_id: edge.id,
            bridge_concept: edge.data?.bridge_concept ?? "Bridge",
          },
          draggable: false,
          selectable: false,
        };
      });

    return [...mapped, ...bridgeMarkers];
  }, [arisEdges, getNodeOpacity, positionedNodes]);

  const flowEdges = useMemo<Edge[]>(() => {
    const nodeById = new Map(positionedNodes.map((node) => [node.id, node]));
    return arisEdges.map((edge) => {
      const sourceNode = nodeById.get(edge.source);
      const targetNode = nodeById.get(edge.target);
      const edgeType = edge.data?.edge_category === "INTER_DOMAIN_BRIDGE" ? "bridge" : "intra_domain";
      return {
        id: edge.id,
        source: edge.source,
        target: edge.target,
        type: edgeType,
        markerEnd: { type: MarkerType.ArrowClosed },
        data: {
          ...edge.data,
          source_cluster_id: sourceNode?.data.cluster_id,
          target_cluster_id: targetNode?.data.cluster_id,
        },
        style: {
          opacity: getEdgeOpacity(edge),
        },
      } as Edge;
    });
  }, [arisEdges, getEdgeOpacity, positionedNodes]);

  const onEdgeClick = (_: React.MouseEvent, edge: Edge) => {
    onSelectEdge(edge.id);
    setActiveBridge(edge.id);
  };

  return (
    <div style={{ width: "100%", height: "100%", background: "#0d0e14" }}>
      <ReactFlow
        nodes={flowNodes}
        edges={flowEdges}
        fitView
        fitViewOptions={{ padding: 0.2 }}
        minZoom={0.18}
        maxZoom={2}
        nodeTypes={nodeTypes}
        edgeTypes={edgeTypes}
        onEdgeClick={onEdgeClick}
        proOptions={{ hideAttribution: true }}
      >
        <Background color="rgba(255,255,255,0.04)" gap={24} size={1} />
        <MiniMap
          nodeColor={minimapNodeColor}
          nodeStrokeWidth={2}
          maskColor="rgba(10,11,20,0.65)"
          position="bottom-right"
          zoomable
          pannable
        />
        <Controls />
      </ReactFlow>
      {activeBridgeEdgeId ? <HypothesisPanel edges={edges} nodes={nodes} /> : null}
    </div>
  );
}
