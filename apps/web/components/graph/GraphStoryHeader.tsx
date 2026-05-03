"use client";

import { useMemo } from "react";

import type { GraphEdge, GraphNode } from "../../lib/api/graphs";

type Props = {
  nodes: GraphNode[];
  edges: GraphEdge[];
  contradictionsCount?: number;
  gapsCount?: number;
};

export function GraphStoryHeader({ nodes, edges, contradictionsCount, gapsCount }: Props) {
  const stats = useMemo(() => {
    const papers = nodes.filter((n) => n.node_type === "document").length;
    const concepts = nodes.filter((n) => n.node_type === "concept").length;
    const domainsSet = new Set<string>();
    for (const n of nodes) {
      if (n.node_type === "concept" && n.cluster_id) {
        domainsSet.add(n.cluster_id);
      }
    }
    const domains = domainsSet.size;
    const bridges = edges.filter((e) => e.edge_type === "cross_domain_bridge").length;
    return { papers, concepts, domains, bridges };
  }, [nodes, edges]);

  if (stats.concepts === 0 && stats.papers === 0) return null;

  const parts: string[] = [];
  if (stats.papers) parts.push(`${stats.papers} paper${stats.papers === 1 ? "" : "s"}`);
  if (stats.concepts) parts.push(`${stats.concepts} concept${stats.concepts === 1 ? "" : "s"}`);
  if (stats.domains) parts.push(`${stats.domains} domain${stats.domains === 1 ? "" : "s"}`);
  if (stats.bridges)
    parts.push(`${stats.bridges} cross-domain bridge${stats.bridges === 1 ? "" : "s"}`);
  if (contradictionsCount && contradictionsCount > 0)
    parts.push(`${contradictionsCount} contradiction${contradictionsCount === 1 ? "" : "s"}`);
  if (gapsCount && gapsCount > 0)
    parts.push(`${gapsCount} research gap${gapsCount === 1 ? "" : "s"}`);

  return (
    <div className="mb-3 flex flex-wrap items-center gap-x-2 gap-y-1 rounded-xl border border-white/8 bg-white/3 px-4 py-2.5 text-[11px] text-ink-2">
      <Icon />
      <span className="font-semibold text-ink">Knowledge graph summary</span>
      <span className="opacity-50">·</span>
      <span>{parts.join(" · ")}</span>
    </div>
  );
}

function Icon() {
  return (
    <svg
      className="h-3.5 w-3.5 text-accent"
      viewBox="0 0 24 24"
      fill="none"
      stroke="currentColor"
      strokeWidth={2}
    >
      <circle cx="12" cy="12" r="3" />
      <circle cx="5" cy="5" r="2" />
      <circle cx="19" cy="5" r="2" />
      <circle cx="5" cy="19" r="2" />
      <circle cx="19" cy="19" r="2" />
      <path strokeLinecap="round" d="M7 6.5l3 4M17 6.5l-3 4M7 17.5l3-4M17 17.5l-3-4" />
    </svg>
  );
}
