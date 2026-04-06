"use client";

import type { CSSProperties } from "react";
import type { NodeProps } from "reactflow";

import { useGraphStore } from "../../lib/stores/graphStore";

const PALETTE = [
  "15,118,110",
  "3,105,161",
  "29,78,216",
  "124,58,237",
  "180,83,9",
  "190,18,60",
  "21,128,61",
  "55,65,81",
];

function colorForCluster(clusterId: string): string {
  let hash = 0;
  for (let i = 0; i < clusterId.length; i += 1) {
    hash = (hash << 5) - hash + clusterId.charCodeAt(i);
    hash |= 0;
  }
  return PALETTE[Math.abs(hash) % PALETTE.length];
}

export function SubDomainNode({ id, data }: NodeProps) {
  const setSelectedNode = useGraphStore((state) => state.setSelectedNode);
  const label = String(data?.label ?? "Subdomain");
  const clusterId = String(data?.cluster_id ?? "General Research");
  const rgb = colorForCluster(clusterId);
  const style: CSSProperties = {
    backgroundColor: `rgba(${rgb},0.30)`,
    border: `1px solid rgba(${rgb},0.55)`,
  };

  return (
    <button
      type="button"
      onClick={() => setSelectedNode(id)}
      className="h-9 w-[90px] rounded-xl px-2 py-1 text-left"
      style={style}
    >
      <span className="block truncate text-[11px] font-semibold text-slate-900">{label}</span>
      <span className="mt-0.5 inline-flex rounded-full bg-white/75 px-1.5 py-0.5 text-[9px] uppercase tracking-wide text-slate-700">
        {clusterId}
      </span>
    </button>
  );
}
