"use client";

import { useMemo } from "react";

import type { GraphEdge, GraphNode, HypothesisStatus } from "../../lib/api/graphs";
import { useHypotheses } from "../../lib/hooks/useHypotheses";
import { useGraphStore } from "../../lib/stores/graphStore";

type HypothesisPanelProps = {
  edges: GraphEdge[];
  nodes: GraphNode[];
};

const STATUS_OPTIONS: HypothesisStatus[] = ["proposed", "investigating", "accepted", "rejected"];

function parseContextFromPathname(): { workspaceId: string | null; graphId: string | null } {
  if (typeof window === "undefined") {
    return { workspaceId: null, graphId: null };
  }

  const parts = window.location.pathname.split("/").filter(Boolean);
  const workspaceIndex = parts.indexOf("workspaces");
  const graphsIndex = parts.indexOf("graphs");

  return {
    workspaceId: workspaceIndex >= 0 ? parts[workspaceIndex + 1] ?? null : null,
    graphId: graphsIndex >= 0 ? parts[graphsIndex + 1] ?? null : null,
  };
}

export function HypothesisPanel({ edges, nodes }: HypothesisPanelProps) {
  const activeBridgeEdgeId = useGraphStore((state) => state.activeBridgeEdgeId);
  const setActiveBridge = useGraphStore((state) => state.setActiveBridge);
  const selectedHypothesisId = useGraphStore((state) => state.selectedHypothesisId);
  const setSelectedHypothesis = useGraphStore((state) => state.setSelectedHypothesis);

  const { workspaceId, graphId } = parseContextFromPathname();
  const { hypotheses, isLoading, updateStatus } = useHypotheses(workspaceId, graphId);

  const activeEdge = useMemo(
    () => edges.find((edge) => edge.id === activeBridgeEdgeId) ?? null,
    [activeBridgeEdgeId, edges],
  );

  const selectedHypothesis = useMemo(() => {
    const byEdge = hypotheses.filter((item) => item.edge_id === activeBridgeEdgeId);
    if (byEdge.length === 0) {
      return null;
    }
    if (!selectedHypothesisId) {
      return byEdge[0];
    }
    return byEdge.find((item) => item.id === selectedHypothesisId) ?? byEdge[0];
  }, [activeBridgeEdgeId, hypotheses, selectedHypothesisId]);

  if (!activeBridgeEdgeId || !activeEdge) {
    return null;
  }

  const source = nodes.find((node) => node.id === activeEdge.source_node_id);
  const target = nodes.find((node) => node.id === activeEdge.target_node_id);

  const sourceLabel = String(source?.label ?? "Source");
  const targetLabel = String(target?.label ?? "Target");
  const bridgeConcept = activeEdge.bridge_concept ?? "Cross-domain bridge";
  const pct = Math.round(activeEdge.confidence * 100);
  const evidenceText = String(activeEdge.evidence?.text ?? "No evidence available");

  const STATUS_COLORS: Record<string, string> = {
    proposed: "border-white/12 bg-white/6 text-ink-2",
    investigating: "border-warning/30 bg-warning/12 text-yellow-300",
    accepted: "border-success/30 bg-success/12 text-emerald-300",
    rejected: "border-danger/30 bg-danger/12 text-red-300",
  };

  return (
    <aside
      className="animate-slide-in absolute right-0 top-0 z-20 flex h-full w-[380px] flex-col border-l border-white/8 shadow-2xl"
      style={{ background: "rgba(17,19,24,0.97)", backdropFilter: "blur(20px)" }}
    >
      {/* Header */}
      <div className="flex items-center justify-between border-b border-white/6 px-5 py-4">
        <div className="flex items-center gap-2">
          <span className="h-2 w-2 rounded-full bg-bridge" />
          <p className="text-[11px] font-semibold uppercase tracking-[0.18em] text-ink-2">Bridge Hypothesis</p>
        </div>
        <button
          type="button"
          className="flex h-7 w-7 items-center justify-center rounded-lg border border-white/10 bg-white/5 text-ink-2 transition hover:bg-white/10 hover:text-ink"
          onClick={() => setActiveBridge(null)}
        >
          <svg className="h-3.5 w-3.5" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}>
            <path strokeLinecap="round" strokeLinejoin="round" d="M6 18L18 6M6 6l12 12"/>
          </svg>
        </button>
      </div>

      <div className="flex-1 overflow-y-auto p-5 space-y-4">
        {/* Bridge concept title */}
        <div>
          <span className="inline-flex rounded-full border border-bridge/30 bg-bridge/12 px-3 py-1 text-sm font-semibold text-orange-300">
            {bridgeConcept}
          </span>
        </div>

        {/* Source → Target */}
        <div className="flex items-center gap-2 text-xs text-ink-2">
          <span className="rounded-lg border border-white/10 bg-white/5 px-2 py-1 font-medium text-ink">{sourceLabel}</span>
          <svg className="h-3 w-3 shrink-0 text-ink-3" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}>
            <path strokeLinecap="round" strokeLinejoin="round" d="M14 5l7 7m0 0l-7 7m7-7H3"/>
          </svg>
          <span className="rounded-lg border border-white/10 bg-white/5 px-2 py-1 font-medium text-ink">{targetLabel}</span>
        </div>

        {/* Confidence */}
        <div>
          <div className="mb-1.5 flex justify-between text-[11px] text-ink-2">
            <span>Confidence</span>
            <span className="font-semibold text-ink">{pct}%</span>
          </div>
          <div className="h-1.5 w-full rounded-full bg-white/8">
            <div className="h-1.5 rounded-full bg-success transition-all" style={{ width: `${pct}%` }} />
          </div>
        </div>

        {/* Hypothesis text */}
        <div className="rounded-xl border-l-2 border-accent bg-accent/6 p-4">
          {isLoading ? (
            <p className="text-sm text-ink-2 animate-pulse">Loading hypothesis…</p>
          ) : (
            <p className="text-sm leading-relaxed text-ink-2 whitespace-pre-line">
              {selectedHypothesis?.hypothesis_text ?? "No hypothesis generated yet for this bridge."}
            </p>
          )}
        </div>

        {/* Status actions */}
        {selectedHypothesis && (
          <div>
            <p className="mb-2 text-[10px] font-semibold uppercase tracking-[0.18em] text-ink-3">Set status</p>
            <div className="flex flex-wrap gap-2">
              {STATUS_OPTIONS.map((statusOption) => {
                const isActive = selectedHypothesis.status === statusOption;
                return (
                  <button
                    key={statusOption}
                    type="button"
                    className={`rounded-full border px-3 py-1 text-xs font-semibold transition ${
                      isActive
                        ? STATUS_COLORS[statusOption] ?? "border-white/12 bg-white/8 text-ink"
                        : "border-white/8 bg-white/3 text-ink-3 hover:border-white/14 hover:text-ink-2"
                    }`}
                    onClick={async () => {
                      setSelectedHypothesis(selectedHypothesis.id);
                      await updateStatus(selectedHypothesis.id, statusOption);
                    }}
                  >
                    {statusOption}
                  </button>
                );
              })}
            </div>
          </div>
        )}

        {/* Evidence */}
        <details className="rounded-xl border border-white/8 bg-white/3">
          <summary className="cursor-pointer select-none px-4 py-3 text-xs font-semibold text-ink-2">
            Evidence chain
          </summary>
          <div className="border-t border-white/6 px-4 py-3 text-xs leading-relaxed text-ink-2">
            {evidenceText}
          </div>
        </details>
      </div>
    </aside>
  );
}
