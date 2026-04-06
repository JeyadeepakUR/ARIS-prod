"use client";

import type { NodeProps } from "reactflow";

import { useGraphStore } from "../../lib/stores/graphStore";

const PALETTE = [
  "#0f766e",
  "#0369a1",
  "#1d4ed8",
  "#7c3aed",
  "#b45309",
  "#be123c",
  "#15803d",
  "#374151",
];

function colorForCluster(clusterId: string): string {
  let hash = 0;
  for (let i = 0; i < clusterId.length; i += 1) {
    hash = (hash << 5) - hash + clusterId.charCodeAt(i);
    hash |= 0;
  }
  return PALETTE[Math.abs(hash) % PALETTE.length];
}

export function DomainNode({ data }: NodeProps) {
  const clusterId = String(data?.cluster_id ?? "General Research");
  const label = String(data?.label ?? "Domain");
  const documentCount = Number(data?.document_count ?? 0);
  const toggleClusterCollapse = useGraphStore((state) => state.toggleClusterCollapse);

  const color = colorForCluster(clusterId);

  return (
    <button
      type="button"
      onClick={() => toggleClusterCollapse(clusterId)}
      className="relative h-12 w-[120px] rounded-full px-3 text-left shadow-md"
      style={{ backgroundColor: color, color: "#ffffff" }}
    >
      <span className="block truncate text-xs font-bold uppercase tracking-wide">{label}</span>
      <span className="absolute -right-1 -top-1 rounded-full bg-white/95 px-1.5 py-0.5 text-[10px] font-semibold text-slate-700">
        {documentCount}
      </span>
    </button>
  );
}
