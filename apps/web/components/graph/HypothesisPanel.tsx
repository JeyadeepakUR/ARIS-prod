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

  const sourceDomain = String(source?.cluster_id ?? "Unknown");
  const targetDomain = String(target?.cluster_id ?? "Unknown");
  const sourceLabel = String(source?.label ?? "Source");
  const targetLabel = String(target?.label ?? "Target");
  const bridgeConcept = activeEdge.bridge_concept ?? "Cross-domain bridge";

  const confidencePct = `${Math.round(activeEdge.confidence * 100)}%`;
  const evidenceText = String(activeEdge.evidence?.text ?? activeEdge.evidence ?? "No evidence available");

  return (
    <aside className="absolute right-0 top-0 z-20 h-full w-[420px] border-l border-slate-200 bg-white/95 p-4 shadow-2xl backdrop-blur-sm">
      <div className="mb-4 flex items-center justify-between">
        <p className="text-xs font-semibold uppercase tracking-[0.18em] text-slate-500">Bridge Hypothesis</p>
        <button
          type="button"
          className="rounded-md border border-slate-300 px-2 py-1 text-xs text-slate-700"
          onClick={() => setActiveBridge(null)}
        >
          Close
        </button>
      </div>

      <p className="text-2xl italic text-slate-800">{bridgeConcept}</p>

      <div className="mt-4 rounded-lg border border-slate-200 bg-slate-50 p-3 text-sm text-slate-700">
        <span>{sourceLabel}</span>
        <span className="mx-2">→</span>
        <span className="rounded-full bg-slate-200 px-2 py-0.5 text-xs">{sourceDomain}</span>
        <span className="mx-2">→</span>
        <span>{targetLabel}</span>
        <span className="mx-2">→</span>
        <span className="rounded-full bg-slate-200 px-2 py-0.5 text-xs">{targetDomain}</span>
      </div>

      <div className="mt-4">
        <div className="mb-1 flex justify-between text-xs text-slate-600">
          <span>Confidence</span>
          <span>{confidencePct}</span>
        </div>
        <div className="h-2 w-full rounded-full bg-slate-200">
          <div className="h-2 rounded-full bg-emerald-500" style={{ width: confidencePct }} />
        </div>
      </div>

      <blockquote className="mt-4 rounded-lg border-l-4 border-indigo-500 bg-indigo-50 p-3 text-sm leading-relaxed text-indigo-900">
        {isLoading ? "Loading hypothesis..." : selectedHypothesis?.hypothesis_text ?? "No hypothesis generated yet for this bridge."}
      </blockquote>

      <div className="mt-4 flex flex-wrap gap-2">
        {STATUS_OPTIONS.map((statusOption) => {
          const active = selectedHypothesis?.status === statusOption;
          return (
            <button
              key={statusOption}
              type="button"
              className={`rounded-full px-2.5 py-1 text-xs font-semibold ${
                active ? "bg-slate-900 text-white" : "border border-slate-300 bg-white text-slate-700"
              }`}
              onClick={async () => {
                if (!selectedHypothesis) {
                  return;
                }
                setSelectedHypothesis(selectedHypothesis.id);
                await updateStatus(selectedHypothesis.id, statusOption);
              }}
            >
              {statusOption}
            </button>
          );
        })}
      </div>

      <details className="mt-4 rounded-lg border border-slate-200 bg-white">
        <summary className="cursor-pointer px-3 py-2 text-sm font-semibold text-slate-700">Evidence</summary>
        <div className="border-t border-slate-200 px-3 py-2 text-sm text-slate-700">{evidenceText}</div>
      </details>
    </aside>
  );
}
