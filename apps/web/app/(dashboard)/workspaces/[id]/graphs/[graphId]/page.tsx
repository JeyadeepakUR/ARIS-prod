"use client";

import Link from "next/link";
import { useParams } from "next/navigation";
import { useEffect, useMemo, useState } from "react";
import { ReactFlowProvider } from "reactflow";
import "reactflow/dist/style.css";

import { EdgeTooltip } from "../../../../../../components/graph/EdgeTooltip";
import { GraphCanvas } from "../../../../../../components/graph/GraphCanvas";
import { GraphControls } from "../../../../../../components/graph/GraphControls";
import { PageHeader } from "../../../../../../components/layout/PageHeader";
import { listGraphEdges, listGraphNodes, type GraphEdge, type GraphNode } from "../../../../../../lib/api/graphs";
import { useGraphStore } from "../../../../../../lib/stores/graphStore";

export default function GraphCanvasPage() {
  const params = useParams<{ id: string; graphId: string }>();
  const workspaceId = params.id;
  const graphId = params.graphId;

  const [nodes, setNodes] = useState<GraphNode[]>([]);
  const [edges, setEdges] = useState<GraphEdge[]>([]);
  const [error, setError] = useState<string | null>(null);
  const [loading, setLoading] = useState(true);
  const [fitTick, setFitTick] = useState(0);
  const [bridgeFocus, setBridgeFocus] = useState(true);

  const {
    selectedEdgeId,
    confidenceThreshold,
    setSelectedEdgeId,
    setConfidenceThreshold,
  } = useGraphStore();

  useEffect(() => {
    let mounted = true;

    async function loadGraph() {
      try {
        const [nodeRows, edgeRows] = await Promise.all([listGraphNodes(graphId), listGraphEdges(graphId)]);
        if (!mounted) {
          return;
        }
        setNodes(nodeRows);
        setEdges(edgeRows);
        setError(null);
      } catch (loadError) {
        if (!mounted) {
          return;
        }
        setError(loadError instanceof Error ? loadError.message : "Unable to load graph");
      } finally {
        if (mounted) {
          setLoading(false);
        }
      }
    }

    loadGraph();
    return () => {
      mounted = false;
    };
  }, [graphId]);

  const selectedEdge = useMemo(
    () => edges.find((edge) => edge.id === selectedEdgeId) ?? null,
    [edges, selectedEdgeId],
  );

  return (
    <div className="-mx-5 -mt-5 md:-mx-7 md:-mt-7 flex flex-col min-h-[calc(100vh-80px)]">
      {/* Top bar */}
      <div className="flex items-center justify-between border-b border-white/6 px-5 py-3 md:px-7">
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
          <span className="text-xs font-mono text-ink-2">{graphId.slice(0,8)}…</span>
          {loading && <span className="text-xs text-ink-3">Loading…</span>}
          {!loading && (
            <span className="rounded-full border border-success/30 bg-success/10 px-2 py-0.5 text-[10px] font-semibold text-emerald-300">
              {nodes.length} nodes · {edges.length} edges
            </span>
          )}
        </div>

        <GraphControls
          confidenceThreshold={confidenceThreshold}
          onChangeThreshold={(value) => setConfidenceThreshold(value)}
          onFit={() => setFitTick((value) => value + 1)}
          bridgeFocus={bridgeFocus}
          onToggleBridgeFocus={() => setBridgeFocus((value) => !value)}
        />
      </div>

      {error ? (
        <div className="mx-5 mt-4 rounded-xl border border-danger/25 bg-danger/8 px-4 py-3 text-sm text-red-300 md:mx-7">
          {error}
        </div>
      ) : null}

      {/* Canvas */}
      <div className="flex-1 relative" style={{ minHeight: 600 }}>
        <ReactFlowProvider>
          <GraphCanvas
            key={fitTick}
            nodes={nodes}
            edges={edges}
            confidenceThreshold={confidenceThreshold}
            bridgeFocus={bridgeFocus}
            onSelectEdge={(edgeId) => setSelectedEdgeId(edgeId)}
          />
        </ReactFlowProvider>
      </div>

      {/* Edge evidence panel */}
      {selectedEdge && (
        <div className="border-t border-white/6 px-5 py-4 md:px-7">
          <p className="mb-2 text-[11px] font-semibold uppercase tracking-[0.18em] text-ink-2">Edge Evidence</p>
          <EdgeTooltip edge={selectedEdge} />
        </div>
      )}
    </div>
  );
}