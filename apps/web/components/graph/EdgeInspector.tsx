"use client";

import { useEffect, useMemo, useState } from "react";

import {
  getChunkEvidence,
  type ChunkEvidence,
  type GraphEdge,
  type GraphNode,
} from "../../lib/api/graphs";
import { colorForDomain } from "../../lib/graph/layout";
import { EvidenceStack } from "./EvidenceCard";

type Props = {
  edge: GraphEdge | null;
  graphId: string;
  allNodes: GraphNode[];
  onClose: () => void;
};

function chunkIdsFromMeta(meta: Record<string, unknown> | undefined | null): string[] {
  const m = meta ?? {};
  const ids = (m.source_chunk_ids ?? m.evidence_chunks ?? m.chunk_ids) as unknown;
  return Array.isArray(ids) ? (ids as string[]).filter(Boolean) : [];
}

export function EdgeInspector({ edge, graphId, allNodes, onClose }: Props) {
  const nodeMap = useMemo(() => new Map(allNodes.map((n) => [n.id, n])), [allNodes]);

  const sourceNode = edge ? nodeMap.get(edge.source_node_id) ?? null : null;
  const targetNode = edge ? nodeMap.get(edge.target_node_id) ?? null : null;

  const sourceChunkIds = useMemo(
    () => chunkIdsFromMeta(sourceNode?.metadata),
    [sourceNode],
  );
  const targetChunkIds = useMemo(
    () => chunkIdsFromMeta(targetNode?.metadata),
    [targetNode],
  );

  const [sourceEvidence, setSourceEvidence] = useState<ChunkEvidence[]>([]);
  const [targetEvidence, setTargetEvidence] = useState<ChunkEvidence[]>([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    let cancelled = false;
    setSourceEvidence([]);
    setTargetEvidence([]);
    setError(null);
    if (!edge) return;
    const merged = Array.from(new Set([...sourceChunkIds, ...targetChunkIds]));
    if (merged.length === 0) return;
    setLoading(true);
    getChunkEvidence(graphId, merged)
      .then((rows) => {
        if (cancelled) return;
        const sourceSet = new Set(sourceChunkIds);
        const targetSet = new Set(targetChunkIds);
        setSourceEvidence(rows.filter((r) => sourceSet.has(r.chunk_id)));
        setTargetEvidence(rows.filter((r) => targetSet.has(r.chunk_id)));
      })
      .catch((err) => {
        if (!cancelled) setError(err instanceof Error ? err.message : "Failed to load evidence");
      })
      .finally(() => {
        if (!cancelled) setLoading(false);
      });
    return () => {
      cancelled = true;
    };
  }, [edge, graphId, sourceChunkIds, targetChunkIds]);

  if (!edge || !sourceNode || !targetNode) return null;

  const isBridge = edge.edge_type === "cross_domain_bridge";
  const sourceColor = colorForDomain(String(sourceNode.cluster_id ?? "General Research"));
  const targetColor = colorForDomain(String(targetNode.cluster_id ?? "General Research"));

  // Bridge evidence is a string like "[A] ...\n[B] ...". Extract reasoning.
  const evidenceText = typeof edge.evidence?.text === "string" ? edge.evidence.text : "";
  const [reasoningA, reasoningB] = (() => {
    const m = evidenceText.match(/\[A\]([\s\S]*?)\[B\]([\s\S]*)/);
    if (m) return [m[1].trim(), m[2].trim()];
    return [evidenceText, ""];
  })();

  return (
    <aside className="fixed right-4 top-24 z-30 flex w-[480px] max-h-[82vh] flex-col rounded-2xl border border-white/10 bg-[#11131a]/95 shadow-2xl backdrop-blur">
      <header
        className="flex items-start justify-between gap-3 border-b border-white/10 p-4"
        style={{
          background: isBridge
            ? "linear-gradient(90deg,rgba(245,158,11,0.18),rgba(245,158,11,0.08))"
            : "rgba(255,255,255,0.04)",
        }}
      >
        <div className="min-w-0">
          <p className="text-[10px] font-semibold uppercase tracking-wider opacity-70">
            {isBridge ? "Cross-domain bridge" : edge.edge_type.replace("_", " ")}
          </p>
          <h3 className="mt-1 text-sm font-bold leading-tight text-ink">
            {edge.bridge_concept ?? `${sourceNode.label} ↔ ${targetNode.label}`}
          </h3>
          <div className="mt-1.5 flex flex-wrap items-center gap-1.5 text-[10px]">
            <span
              className="rounded-full border px-2 py-0.5 font-semibold"
              style={{ borderColor: `${sourceColor}55`, color: sourceColor }}
            >
              <span
                className="mr-1 inline-block h-1.5 w-1.5 rounded-full align-middle"
                style={{ background: sourceColor }}
              />
              {sourceNode.cluster_id ?? "General"}
            </span>
            <span className="text-ink-3">↔</span>
            <span
              className="rounded-full border px-2 py-0.5 font-semibold"
              style={{ borderColor: `${targetColor}55`, color: targetColor }}
            >
              <span
                className="mr-1 inline-block h-1.5 w-1.5 rounded-full align-middle"
                style={{ background: targetColor }}
              />
              {targetNode.cluster_id ?? "General"}
            </span>
          </div>
        </div>
        <button
          type="button"
          onClick={onClose}
          aria-label="Close"
          className="rounded-full p-1 text-ink-3 hover:bg-white/10 hover:text-ink"
        >
          <svg className="h-4 w-4" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth={2}>
            <path strokeLinecap="round" strokeLinejoin="round" d="M6 6l12 12M6 18L18 6" />
          </svg>
        </button>
      </header>

      <div className="flex-1 space-y-4 overflow-y-auto p-4 text-xs text-ink-2">
        <div className="flex items-center gap-2">
          <span className="opacity-70">Bridge confidence:</span>
          <span className="font-mono font-semibold text-ink">
            {(edge.confidence * 100).toFixed(0)}%
          </span>
          <div className="h-1.5 flex-1 overflow-hidden rounded-full bg-white/5">
            <div
              className="h-full rounded-full bg-amber-400"
              style={{ width: `${edge.confidence * 100}%` }}
            />
          </div>
        </div>

        {(reasoningA || reasoningB) && (
          <Section label="Why these connect" hint="LLM bridge reasoning">
            {reasoningA && (
              <p className="mb-1.5 rounded-lg border border-white/10 bg-white/5 px-3 py-2 text-[11px] leading-relaxed">
                <span className="mr-1 font-semibold" style={{ color: sourceColor }}>
                  A:
                </span>
                {reasoningA}
              </p>
            )}
            {reasoningB && (
              <p className="rounded-lg border border-white/10 bg-white/5 px-3 py-2 text-[11px] leading-relaxed">
                <span className="mr-1 font-semibold" style={{ color: targetColor }}>
                  B:
                </span>
                {reasoningB}
              </p>
            )}
          </Section>
        )}

        <Section label="Evidence side by side" hint="Quoted text from each paper">
          {loading && <p className="text-[10px] italic text-ink-3">Loading evidence…</p>}
          {error && <p className="text-[10px] italic text-red-300">{error}</p>}
          {!loading && !error && (
            <div className="grid grid-cols-1 gap-3">
              <div>
                <p className="mb-1.5 flex items-center gap-1.5 text-[10px] font-semibold uppercase tracking-wider">
                  <span
                    className="h-1.5 w-1.5 rounded-full"
                    style={{ background: sourceColor }}
                  />
                  <span style={{ color: sourceColor }}>{sourceNode.label}</span>
                </p>
                <EvidenceStack
                  evidence={sourceEvidence}
                  accent={sourceColor}
                  emptyText="No source-side excerpts stored."
                />
              </div>
              <div>
                <p className="mb-1.5 flex items-center gap-1.5 text-[10px] font-semibold uppercase tracking-wider">
                  <span
                    className="h-1.5 w-1.5 rounded-full"
                    style={{ background: targetColor }}
                  />
                  <span style={{ color: targetColor }}>{targetNode.label}</span>
                </p>
                <EvidenceStack
                  evidence={targetEvidence}
                  accent={targetColor}
                  emptyText="No target-side excerpts stored."
                />
              </div>
            </div>
          )}
        </Section>
      </div>
    </aside>
  );
}

function Section({
  label,
  hint,
  children,
}: {
  label: string;
  hint?: string;
  children: React.ReactNode;
}) {
  return (
    <div>
      <div className="mb-1.5 flex items-baseline justify-between gap-2">
        <p className="text-[10px] font-semibold uppercase tracking-[0.18em] text-ink-3">{label}</p>
        {hint && <p className="text-[9px] italic text-ink-3 opacity-70">{hint}</p>}
      </div>
      {children}
    </div>
  );
}
