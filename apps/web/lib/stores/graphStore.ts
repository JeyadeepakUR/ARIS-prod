import { create } from "zustand";

type GraphStore = {
  // Edge selection
  selectedEdgeId: string | null;
  setSelectedEdgeId: (id: string | null) => void;

  // Active bridge edge (hover / click in BridgeEdge)
  activeBridgeEdgeId: string | null;
  setActiveBridge: (id: string | null) => void;

  // Node selection
  selectedNodeId: string | null;
  setSelectedNode: (id: string | null) => void;

  // Hypothesis panel
  selectedHypothesisId: string | null;
  setSelectedHypothesis: (id: string | null) => void;

  // Confidence threshold filter
  confidenceThreshold: number;
  setConfidenceThreshold: (value: number) => void;

  // Bridge focus mode (highlight cross-domain edges)
  bridgeFocusMode: boolean;
  toggleBridgeFocus: () => void;

  // Collapsed clusters
  collapsedClusters: Set<string>;
  toggleClusterCollapse: (clusterId: string) => void;

  // Opacity helpers
  getNodeOpacity: (node: { id: string; data: { cluster_id?: string } }) => number;
  getEdgeOpacity: (edge: { data?: { confidence?: number } }) => number;
};

export const useGraphStore = create<GraphStore>((set, get) => ({
  selectedEdgeId: null,
  setSelectedEdgeId: (id) => set({ selectedEdgeId: id }),

  activeBridgeEdgeId: null,
  setActiveBridge: (id) => set({ activeBridgeEdgeId: id }),

  selectedNodeId: null,
  setSelectedNode: (id) => set({ selectedNodeId: id }),

  selectedHypothesisId: null,
  setSelectedHypothesis: (id) => set({ selectedHypothesisId: id }),

  confidenceThreshold: 0.5,
  setConfidenceThreshold: (value) => set({ confidenceThreshold: value }),

  bridgeFocusMode: false,
  toggleBridgeFocus: () => set((s) => ({ bridgeFocusMode: !s.bridgeFocusMode })),

  collapsedClusters: new Set(),
  toggleClusterCollapse: (clusterId) =>
    set((s) => {
      const next = new Set(s.collapsedClusters);
      if (next.has(clusterId)) next.delete(clusterId);
      else next.add(clusterId);
      return { collapsedClusters: next };
    }),

  getNodeOpacity: (node) => {
    const { collapsedClusters } = get();
    const clusterId = node.data?.cluster_id ?? "";
    return collapsedClusters.has(clusterId) ? 0.15 : 1;
  },

  getEdgeOpacity: (edge) => {
    const { confidenceThreshold } = get();
    const confidence = edge.data?.confidence ?? 1;
    return confidence >= confidenceThreshold ? 1 : 0.1;
  },
}));
