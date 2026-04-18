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
    confidence: number;
    bridge_concept?: string;
    evidence?: string;
  };
};

const TIER_Y: Record<number, number> = { 1: 0, 2: 200, 3: 400 };
const CLUSTER_SPACING = 320;
const NODE_SPACING = 160;

/**
 * Deterministic radial-cluster layout.
 * Tier-1 (domain) nodes form the top row, tier-2 (subdomain) in the middle,
 * tier-3 (concept) at the bottom. Within each tier nodes are spread by cluster.
 */
export function computeGraphLayout(nodes: ARISNode[], _edges: ARISEdge[]): ARISNode[] {
  if (nodes.length === 0) return [];

  // Group nodes by tier then cluster
  const byTierCluster: Record<number, Record<string, ARISNode[]>> = { 1: {}, 2: {}, 3: {} };

  for (const node of nodes) {
    const tier = node.data.tier ?? 3;
    const cluster = node.data.cluster_id ?? "General";
    if (!byTierCluster[tier]) byTierCluster[tier] = {};
    if (!byTierCluster[tier][cluster]) byTierCluster[tier][cluster] = [];
    byTierCluster[tier][cluster].push(node);
  }

  const positioned: ARISNode[] = [];

  for (const [tierStr, clusters] of Object.entries(byTierCluster)) {
    const tier = Number(tierStr);
    const baseY = TIER_Y[tier] ?? tier * 200;
    const clusterNames = Object.keys(clusters).sort();

    clusterNames.forEach((cluster, clusterIdx) => {
      const clusterNodes = clusters[cluster];
      const clusterBaseX = clusterIdx * CLUSTER_SPACING;

      clusterNodes.forEach((node, nodeIdx) => {
        const col = nodeIdx % 4;
        const row = Math.floor(nodeIdx / 4);
        positioned.push({
          ...node,
          position: {
            x: clusterBaseX + col * NODE_SPACING,
            y: baseY + row * NODE_SPACING,
          },
        });
      });
    });
  }

  return positioned;
}
