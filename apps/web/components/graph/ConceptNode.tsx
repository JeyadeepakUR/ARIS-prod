"use client";

import type { NodeProps } from "reactflow";
import { Handle, Position, useViewport } from "reactflow";

import { colorForDomain } from "../../lib/graph/layout";

export function ConceptNode({ data }: NodeProps) {
  const viewport = useViewport();
  const label = String(data?.label ?? "Concept");
  const cluster = String(data?.cluster_id ?? "General Research");
  const lowValue = Boolean(data?.low_value);
  const color = colorForDomain(cluster);

  const zoomedOut = viewport.zoom <= 0.4;
  const opacity = lowValue ? 0.35 : zoomedOut ? 0.75 : 1;

  return (
    <div
      className="group relative inline-flex max-w-[200px] items-center gap-1.5 rounded-full border px-2.5 py-1 text-[11px] font-medium shadow-sm transition"
      style={{
        opacity,
        background: `${color}1a`,
        borderColor: `${color}55`,
        color: "#e5e7eb",
      }}
      title={`${label} · ${cluster}`}
    >
      <Handle type="target" position={Position.Left} style={{ background: "transparent", border: "none" }} />
      <span
        className="h-1.5 w-1.5 flex-shrink-0 rounded-full"
        style={{ background: color, boxShadow: `0 0 6px ${color}99` }}
      />
      <span className="truncate">{label}</span>
      <Handle type="source" position={Position.Right} style={{ background: "transparent", border: "none" }} />
    </div>
  );
}
