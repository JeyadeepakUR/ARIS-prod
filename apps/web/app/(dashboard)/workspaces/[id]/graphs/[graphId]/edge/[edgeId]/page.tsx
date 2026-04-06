"use client";

import Link from "next/link";
import { useParams } from "next/navigation";
import { useEffect, useState } from "react";

import { EdgeTooltip } from "../../../../../../../../components/graph/EdgeTooltip";
import { PageHeader } from "../../../../../../../../components/layout/PageHeader";
import { listGraphEdges, type GraphEdge } from "../../../../../../../../lib/api/graphs";

export default function EdgeEvidencePage() {
  const params = useParams<{ id: string; graphId: string; edgeId: string }>();
  const workspaceId = params.id;
  const graphId = params.graphId;
  const edgeId = params.edgeId;

  const [edge, setEdge] = useState<GraphEdge | null>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    let mounted = true;
    listGraphEdges(graphId)
      .then((edges) => {
        if (!mounted) {
          return;
        }
        setEdge(edges.find((item) => item.id === edgeId) ?? null);
      })
      .catch((loadError) => {
        if (!mounted) {
          return;
        }
        setError(loadError instanceof Error ? loadError.message : "Unable to load edge");
      });

    return () => {
      mounted = false;
    };
  }, [edgeId, graphId]);

  return (
    <div>
      <PageHeader
        eyebrow="Edge Detail"
        title="Evidence Chain"
        description="Deep-dive on the selected connection and confidence score."
      />

      {error ? <p className="mb-3 text-sm text-red-700">{error}</p> : null}
      {edge ? (
        <EdgeTooltip edge={edge} />
      ) : (
        <p className="rounded-xl border border-dashed border-spice/30 bg-white/70 p-4 text-sm text-ink/70">
          Edge not found.
        </p>
      )}

      <div className="mt-4 flex gap-2">
        <Link
          href={`/workspaces/${workspaceId}/graphs/${graphId}`}
          className="rounded-lg border border-spice/25 bg-white px-3 py-2 text-xs font-semibold uppercase tracking-wide text-spice"
        >
          Back to Canvas
        </Link>
        <Link
          href={`/workspaces/${workspaceId}/graphs`}
          className="rounded-lg bg-pine px-3 py-2 text-xs font-semibold uppercase tracking-wide text-white"
        >
          Graph List
        </Link>
      </div>
    </div>
  );
}