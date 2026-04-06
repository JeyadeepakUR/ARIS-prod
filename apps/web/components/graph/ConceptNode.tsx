"use client";

import type { CSSProperties } from "react";
import type { NodeProps } from "reactflow";
import { useViewport } from "reactflow";

export function ConceptNode({ data }: NodeProps) {
  const viewport = useViewport();
  const label = String(data?.label ?? "Concept");
  const lowValue = Boolean(data?.low_value);

  if (viewport.zoom <= 1.2) {
    return null;
  }

  const style: CSSProperties = {
    opacity: lowValue ? 0.4 : 1,
    pointerEvents: lowValue ? "none" : "auto",
  };

  return (
    <div
      className="inline-flex min-h-6 max-w-[220px] items-center rounded-full border border-[#e0e0e0] bg-white px-2.5 py-0.5 text-[11px] font-medium text-slate-700"
      style={style}
    >
      <span className="truncate">{label}</span>
    </div>
  );
}
