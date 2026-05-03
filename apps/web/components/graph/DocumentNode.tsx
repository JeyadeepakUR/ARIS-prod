"use client";

import type { NodeProps } from "reactflow";
import { Handle, Position } from "reactflow";

export function DocumentNode({ data }: NodeProps) {
  const label = String(data?.label ?? "Document");

  return (
    <div
      className="flex items-center gap-2.5 rounded-xl border px-3 py-2 text-left shadow-sm"
      style={{
        background: "rgba(148, 163, 184, 0.12)",
        borderColor: "rgba(148, 163, 184, 0.45)",
        color: "#e2e8f0",
        maxWidth: 220,
      }}
      title={label}
    >
      <Handle type="target" position={Position.Left} style={{ background: "transparent", border: "none" }} />
      <svg className="h-4 w-4 flex-shrink-0 opacity-70" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth={2}>
        <path strokeLinecap="round" strokeLinejoin="round" d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z" />
        <path strokeLinecap="round" strokeLinejoin="round" d="M14 2v6h6M16 13H8M16 17H8M10 9H8" />
      </svg>
      <span className="text-[11px] font-medium leading-tight truncate">{label}</span>
      <Handle type="source" position={Position.Right} style={{ background: "transparent", border: "none" }} />
    </div>
  );
}
