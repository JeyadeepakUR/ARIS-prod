"use client";

import Link from "next/link";
import { useParams } from "next/navigation";
import { useEffect, useMemo, useState } from "react";
import { ReactFlowProvider } from "reactflow";
import "reactflow/dist/style.css";

import { BridgeExplorer } from "../../../../../../components/graph/BridgeExplorer";
import { ContradictionsPanel } from "../../../../../../components/graph/ContradictionsPanel";
import { EdgeTooltip } from "../../../../../../components/graph/EdgeTooltip";
import { GapPanel } from "../../../../../../components/graph/GapPanel";
import { GraphCanvas } from "../../../../../../components/graph/GraphCanvas";
import { GraphControls } from "../../../../../../components/graph/GraphControls";
import { GraphStoryHeader } from "../../../../../../components/graph/GraphStoryHeader";
import type { ViewMode } from "../../../../../../lib/graph/layout";
import {
  getGraph,
  listGraphContradictions,
  listGraphEdges,
  listGraphNodes,
  listGraphPlans,
  type Contradiction,
  type GraphEdge,
  type GraphNode,
  type PlanAction,
} from "../../../../../../lib/api/graphs";
import { useHypotheses } from "../../../../../../lib/hooks/useHypotheses";
import { useGraphStore } from "../../../../../../lib/stores/graphStore";

type Tab = "graph" | "bridges" | "contradictions" | "gaps";

export default function GraphCanvasPage() {
  const params = useParams<{ id: string; graphId: string }>();
  const workspaceId = params.id;
  const graphId = params.graphId;

  const [tab, setTab] = useState<Tab>("graph");
  const [nodes, setNodes] = useState<GraphNode[]>([]);
  const [edges, setEdges] = useState<GraphEdge[]>([]);
  const [contradictions, setContradictions] = useState<Contradiction[]>([]);
  const [gaps, setGaps] = useState<PlanAction[]>([]);
  const [error, setError] = useState<string | null>(null);
  const [loading, setLoading] = useState(true);
  const [contradictionsLoading, setContradictionsLoading] = useState(false);
  const [gapsLoading, setGapsLoading] = useState(false);
  const [fitTick, setFitTick] = useState(0);
  const [bridgeFocus, setBridgeFocus] = useState(false);
  const [viewMode, setViewMode] = useState<ViewMode>("by-domain");
  const [graphStatus, setGraphStatus] = useState<string | null>(null);

  const { selectedEdgeId, confidenceThreshold, setSelectedEdgeId, setConfidenceThreshold } =
    useGraphStore();

  // Hypotheses for the Bridges tab — only fetched when that tab is first opened
  const hypothesesEnabled = tab === "bridges";
  const { hypotheses, isLoading: hypothesesLoading, updateStatus } = useHypotheses(
    hypothesesEnabled ? workspaceId : null,
    hypothesesEnabled ? graphId : null,
  );

  async function reloadGraphData() {
    const [nodeRows, edgeRows] = await Promise.all([
      listGraphNodes(graphId),
      listGraphEdges(graphId),
    ]);
    setNodes(nodeRows);
    setEdges(edgeRows);
    setError(null);
  }

  // Load graph canvas data on mount; poll if the graph is still building
  useEffect(() => {
    let mounted = true;
    let pollTimer: ReturnType<typeof setTimeout> | null = null;

    async function loadGraph() {
      try {
        const [graph, nodeRows, edgeRows] = await Promise.all([
          getGraph(workspaceId, graphId),
          listGraphNodes(graphId),
          listGraphEdges(graphId),
        ]);
        if (!mounted) return;
        setGraphStatus(graph.status);
        setNodes(nodeRows);
        setEdges(edgeRows);
        setError(null);

        // If build is still in progress, poll until complete
        if (graph.status === "processing" || graph.status === "pending") {
          pollTimer = setTimeout(loadGraph, 5000);
        }
      } catch (loadError) {
        if (!mounted) return;
        setError(loadError instanceof Error ? loadError.message : "Unable to load graph");
      } finally {
        if (mounted) setLoading(false);
      }
    }
    loadGraph();
    return () => {
      mounted = false;
      if (pollTimer) clearTimeout(pollTimer);
    };
  }, [graphId, workspaceId]);

  // Lazy-load contradictions on first visit to that tab
  useEffect(() => {
    if (tab !== "contradictions" || contradictions.length > 0 || contradictionsLoading) return;
    setContradictionsLoading(true);
    listGraphContradictions(graphId)
      .then(setContradictions)
      .catch(() => setContradictions([]))
      .finally(() => setContradictionsLoading(false));
  }, [tab, graphId, contradictions.length, contradictionsLoading]);

  // Lazy-load gaps on first visit to that tab
  useEffect(() => {
    if (tab !== "gaps" || gaps.length > 0 || gapsLoading) return;
    setGapsLoading(true);
    listGraphPlans(graphId)
      .then(setGaps)
      .catch(() => setGaps([]))
      .finally(() => setGapsLoading(false));
  }, [tab, graphId, gaps.length, gapsLoading]);

  const selectedEdge = useMemo(
    () => edges.find((edge) => edge.id === selectedEdgeId) ?? null,
    [edges, selectedEdgeId],
  );

  const bridgeEdges = useMemo(
    () => edges.filter((e) => e.edge_type === "cross_domain_bridge"),
    [edges],
  );

  const TABS: { id: Tab; label: string; count?: number }[] = [
    { id: "graph", label: "Graph" },
    { id: "bridges", label: "Bridges", count: bridgeEdges.length || undefined },
    { id: "contradictions", label: "Contradictions", count: contradictions.length || undefined },
    { id: "gaps", label: "Gaps", count: gaps.length || undefined },
  ];

  return (
    <div>
      {/* Breadcrumb */}
      <div className="mb-4 flex flex-wrap items-center justify-between gap-3">
        <div className="flex items-center gap-3">
          <Link
            href={`/workspaces/${workspaceId}/graphs`}
            className="flex items-center gap-1.5 text-xs text-ink-2 hover:text-ink transition"
          >
            <svg className="h-3.5 w-3.5" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}>
              <path strokeLinecap="round" strokeLinejoin="round" d="M15 19l-7-7 7-7"/>
            </svg>
            Graphs
          </Link>
          <span className="text-ink-3">/</span>
          <span className="text-xs font-mono text-ink-2">{graphId.slice(0, 8)}…</span>
          {loading && <span className="text-xs text-ink-3">Loading…</span>}
          {!loading && (
            <span className="rounded-full border border-success/30 bg-success/10 px-2 py-0.5 text-[10px] font-semibold text-emerald-300">
              {nodes.length} nodes · {edges.length} edges
            </span>
          )}
          {graphStatus === "processing" && (
            <span className="text-[10px] text-amber-400 animate-pulse">Building…</span>
          )}
          <button
            type="button"
            onClick={() => reloadGraphData().catch(() => undefined)}
            title="Refresh graph data"
            className="ml-1 rounded-lg border border-white/10 bg-white/5 px-2 py-0.5 text-[10px] text-ink-2 hover:bg-white/10 hover:text-ink transition"
          >
            ↻ Refresh
          </button>
        </div>

        {tab === "graph" && (
          <GraphControls
            confidenceThreshold={confidenceThreshold}
            onChangeThreshold={(value) => setConfidenceThreshold(value)}
            onFit={() => setFitTick((value) => value + 1)}
            bridgeFocus={bridgeFocus}
            onToggleBridgeFocus={() => setBridgeFocus((value) => !value)}
            viewMode={viewMode}
            onChangeViewMode={(mode) => setViewMode(mode)}
          />
        )}
      </div>

      {error && (
        <div className="mb-4 rounded-xl border border-danger/25 bg-danger/8 px-4 py-3 text-sm text-red-300">
          {error}
        </div>
      )}

      {/* Tab bar */}
      <div className="mb-4 flex gap-1 rounded-xl border border-white/8 bg-white/3 p-1 w-fit">
        {TABS.map(({ id, label, count }) => (
          <button
            key={id}
            type="button"
            onClick={() => setTab(id)}
            className={`flex items-center gap-1.5 rounded-lg px-4 py-1.5 text-xs font-semibold transition ${
              tab === id
                ? "bg-white/10 text-ink"
                : "text-ink-2 hover:text-ink hover:bg-white/5"
            }`}
          >
            {label}
            {count !== undefined && (
              <span className={`rounded-full px-1.5 py-0.5 text-[10px] font-bold ${
                tab === id ? "bg-accent/20 text-accent-2" : "bg-white/8 text-ink-3"
              }`}>
                {count}
              </span>
            )}
          </button>
        ))}
      </div>

      {/* Tab content */}
      {tab === "graph" && (
        <>
          <GraphStoryHeader
            nodes={nodes}
            edges={edges}
            contradictionsCount={contradictions.length || undefined}
            gapsCount={gaps.length || undefined}
          />
          <div className="overflow-hidden rounded-2xl border border-white/8" style={{ height: 620 }}>
            <ReactFlowProvider>
              <GraphCanvas
                key={fitTick}
                graphId={graphId}
                nodes={nodes}
                edges={edges}
                confidenceThreshold={confidenceThreshold}
                bridgeFocus={bridgeFocus}
                viewMode={viewMode}
                onSelectEdge={(edgeId) => setSelectedEdgeId(edgeId)}
              />
            </ReactFlowProvider>
          </div>

          {selectedEdge && (
            <div className="mt-4">
              <p className="mb-2 text-[11px] font-semibold uppercase tracking-[0.18em] text-ink-2">
                Edge Evidence
              </p>
              <EdgeTooltip edge={selectedEdge} />
            </div>
          )}
        </>
      )}

      {tab === "bridges" && (
        <div className="glass rounded-2xl p-5">
          <div className="mb-5 flex items-baseline justify-between">
            <p className="text-xs font-semibold uppercase tracking-[0.18em] text-ink-2">
              Cross-domain bridges
            </p>
            <p className="text-[11px] text-ink-3">
              {bridgeEdges.length} bridge{bridgeEdges.length !== 1 ? "s" : ""} · {hypotheses.length} hypothes{hypotheses.length !== 1 ? "es" : "is"}
            </p>
          </div>
          <BridgeExplorer
            bridges={bridgeEdges}
            nodes={nodes}
            hypotheses={hypotheses}
            hypothesesLoading={hypothesesLoading}
            onUpdateStatus={updateStatus}
          />
        </div>
      )}

      {tab === "contradictions" && (
        <div className="glass rounded-2xl p-5">
          <p className="mb-4 text-xs font-semibold uppercase tracking-[0.18em] text-ink-2">
            Contradictions detected
          </p>
          <ContradictionsPanel
            contradictions={contradictions}
            loading={contradictionsLoading}
            graphId={graphId}
          />
        </div>
      )}

      {tab === "gaps" && (
        <div className="glass rounded-2xl p-5">
          <p className="mb-4 text-xs font-semibold uppercase tracking-[0.18em] text-ink-2">
            Research gaps &amp; investigation actions
          </p>
          <GapPanel actions={gaps} loading={gapsLoading} graphId={graphId} nodes={nodes} />
        </div>
      )}
    </div>
  );
}
