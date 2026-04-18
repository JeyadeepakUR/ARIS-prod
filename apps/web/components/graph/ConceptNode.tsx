"use client";

import type { NodeProps } from "reactflow";
import { Handle, Position, useViewport } from "reactflow";

export function ConceptNode({ data }: NodeProps) {
  const viewport = useViewport();
  const label = String(data?.label ?? "Concept");
  const lowValue = Boolean(data?.low_value);

  if (viewport.zoom <= 1.1) {
    return null;
  }

  return (
    <div
      className="inline-flex max-w-[180px] items-center rounded-full border border-white/10 bg-white/5 px-2.5 py-0.5 text-[11px] text-ink-2"
      style={{ opacity: lowValue ? 0.3 : 0.85 }}
    >
      <Handle type="target" position={Position.Left} style={{ background: "transparent", border: "none" }} />
      <span className="truncate">{label}</span>
      <Handle type="source" position={Position.Right} style={{ background: "transparent", border: "none" }} />
    </div>
  );
}
