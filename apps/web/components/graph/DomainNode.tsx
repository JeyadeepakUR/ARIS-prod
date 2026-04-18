"use client";

import type { NodeProps } from "reactflow";
import { Handle, Position } from "reactflow";

import { useGraphStore } from "../../lib/stores/graphStore";

const PALETTE = [
  { bg: "rgba(99,102,241,0.18)", border: "rgba(99,102,241,0.5)", text: "#a5b4fc" },
  { bg: "rgba(20,184,166,0.18)", border: "rgba(20,184,166,0.5)", text: "#5eead4" },
  { bg: "rgba(245,158,11,0.18)", border: "rgba(245,158,11,0.5)", text: "#fcd34d" },
  { bg: "rgba(239,68,68,0.18)", border: "rgba(239,68,68,0.5)", text: "#fca5a5" },
  { bg: "rgba(168,85,247,0.18)", border: "rgba(168,85,247,0.5)", text: "#d8b4fe" },
  { bg: "rgba(34,197,94,0.18)", border: "rgba(34,197,94,0.5)", text: "#86efac" },
  { bg: "rgba(14,165,233,0.18)", border: "rgba(14,165,233,0.5)", text: "#7dd3fc" },
  { bg: "rgba(249,115,22,0.18)", border: "rgba(249,115,22,0.5)", text: "#fdba74" },
];

function paletteForCluster(clusterId: string) {
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
  const toggleClusterCollapse = useGraphStore((state) => state.toggleClusterCollapse);
  const p = paletteForCluster(clusterId);

  return (
    <button
      type="button"
      onClick={() => toggleClusterCollapse(clusterId)}
      className="flex items-center gap-2 rounded-2xl px-4 py-2.5 text-left shadow-lg"
      style={{ background: p.bg, border: `1.5px solid ${p.border}` }}
    >
      <Handle type="target" position={Position.Left} style={{ background: p.border, border: "none", width: 8, height: 8 }} />
      <span className="block text-sm font-bold leading-tight" style={{ color: p.text }}>
        {label}
      </span>
      <Handle type="source" position={Position.Right} style={{ background: p.border, border: "none", width: 8, height: 8 }} />
    </button>
  );
}
