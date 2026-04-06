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
    <div className="md:-mx-2 xl:-mx-4">
      <PageHeader
        eyebrow="Graph Canvas"
        title="Knowledge Graph View"
        description="Click any edge to inspect its reasoning chain and confidence score."
      />

      <GraphControls
        confidenceThreshold={confidenceThreshold}
        onChangeThreshold={(value) => setConfidenceThreshold(value)}
        onFit={() => setFitTick((value) => value + 1)}
        bridgeFocus={bridgeFocus}
        onToggleBridgeFocus={() => setBridgeFocus((value) => !value)}
      />

      {error ? <p className="mb-3 text-sm text-red-700">{error}</p> : null}
      {loading ? <p className="mb-3 text-sm text-ink/70">Loading graph...</p> : null}

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

      <div className="mt-4 grid gap-4 xl:grid-cols-[1fr_auto]">
        <div>
          <p className="mb-2 text-xs font-semibold uppercase tracking-[0.2em] text-spice">Edge Evidence</p>
          {selectedEdge ? (
            <EdgeTooltip edge={selectedEdge} />
          ) : (
            <p className="rounded-xl border border-dashed border-spice/30 bg-white/70 p-4 text-sm text-ink/70">
              Select an edge on the canvas to inspect full evidence.
            </p>
          )}
        </div>

        <div className="flex flex-col gap-2">
          {selectedEdge ? (
            <Link
              href={`/workspaces/${workspaceId}/graphs/${graphId}/edge/${selectedEdge.id}`}
              className="rounded-lg bg-spice px-3 py-2 text-xs font-semibold uppercase tracking-wide text-white"
            >
              Open Edge Page
            </Link>
          ) : null}
          <Link
            href={`/workspaces/${workspaceId}/graphs`}
            className="rounded-lg border border-spice/25 bg-white px-3 py-2 text-xs font-semibold uppercase tracking-wide text-spice"
          >
            Back to Graphs
          </Link>
        </div>
      </div>
    </div>
  );
}