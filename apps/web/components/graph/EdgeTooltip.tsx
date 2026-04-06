import type { GraphEdge } from "../../lib/api/graphs";

type EdgeTooltipProps = {
  edge: GraphEdge;
};

export function EdgeTooltip({ edge }: EdgeTooltipProps) {
  const evidenceText = typeof edge.evidence.text === "string" ? edge.evidence.text : "No evidence text";
  return (
    <div className="rounded-xl border border-spice/20 bg-white p-4">
      <div className="mb-2 flex items-center justify-between gap-3">
        <p className="text-sm font-semibold text-ink">{edge.edge_type}</p>
        <span className="rounded-full bg-amber-100 px-2 py-0.5 text-xs font-semibold text-amber-900">
          {(edge.confidence * 100).toFixed(0)}%
        </span>
      </div>
      <p className="text-xs uppercase tracking-wide text-spice">Reasoning chain</p>
      <p className="mt-1 text-sm leading-relaxed text-ink/85">{evidenceText}</p>
    </div>
  );
}