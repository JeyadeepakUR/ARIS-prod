"use client";

import type { NodeProps } from "reactflow";
import { Handle, Position } from "reactflow";

import { useGraphStore } from "../../lib/stores/graphStore";

export function SubDomainNode({ id, data }: NodeProps) {
  const setSelectedNode = useGraphStore((state) => state.setSelectedNode);
  const label = String(data?.label ?? "Subdomain");

  return (
    <button
      type="button"
      onClick={() => setSelectedNode(id)}
      className="rounded-xl border border-white/12 bg-white/6 px-3 py-2 text-left shadow transition hover:bg-white/10"
    >
      <Handle type="target" position={Position.Left} style={{ background: "#555870", border: "none", width: 6, height: 6 }} />
      <span className="block max-w-[110px] truncate text-[12px] font-semibold text-ink-2">{label}</span>
      <Handle type="source" position={Position.Right} style={{ background: "#555870", border: "none", width: 6, height: 6 }} />
    </button>
  );
}
