"use client";

import type { GraphEdge } from "../../lib/api/graphs";

type EdgeTooltipProps = {
  edge: GraphEdge;
};

export function EdgeTooltip({ edge }: EdgeTooltipProps) {
  const evidenceText = typeof edge.evidence?.text === "string" ? edge.evidence.text : "No evidence text";
  const isBridge = edge.edge_category === "INTER_DOMAIN_BRIDGE";
  const pct = Math.round(edge.confidence * 100);

  return (
    <div className="rounded-xl border border-white/8 bg-bg-3 p-4">
      <div className="mb-3 flex items-center justify-between gap-3">
        <div className="flex items-center gap-2">
          <span className={`inline-flex rounded-full px-2.5 py-0.5 text-[10px] font-bold uppercase tracking-wider ${
            isBridge ? "bg-bridge/20 text-orange-300 border border-bridge/30" : "bg-white/8 text-ink-2 border border-white/10"
          }`}>
            {edge.edge_type.replace(/_/g," ")}
          </span>
          {edge.bridge_concept && (
            <span className="text-xs text-ink-2">· {edge.bridge_concept}</span>
          )}
        </div>
        <div className="flex items-center gap-2">
          <div className="h-1.5 w-16 rounded-full bg-white/8">
            <div className="h-1.5 rounded-full bg-success" style={{ width: `${pct}%` }} />
          </div>
          <span className="text-xs font-semibold text-ink-2">{pct}%</span>
        </div>
      </div>
      <p className="mb-1 text-[10px] font-semibold uppercase tracking-[0.18em] text-ink-3">Reasoning chain</p>
      <p className="text-sm leading-relaxed text-ink-2">{evidenceText}</p>
    </div>
  );
}