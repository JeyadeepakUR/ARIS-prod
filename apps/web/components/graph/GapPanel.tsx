"use client";

import { useEffect, useMemo, useState } from "react";

import {
  getChunkEvidence,
  type ChunkEvidence,
  type GraphNode,
  type PlanAction,
} from "../../lib/api/graphs";
import { colorForDomain } from "../../lib/graph/layout";
import { EvidenceCard } from "./EvidenceCard";

const PRIORITY_LABEL = (p: number) => {
  if (p >= 0.75) return { label: "High", style: "text-amber-300 border-amber-500/30 bg-amber-500/8" };
  if (p >= 0.4) return { label: "Medium", style: "text-sky-300 border-sky-500/30 bg-sky-500/8" };
  return { label: "Low", style: "text-ink-2 border-white/10 bg-white/4" };
};

const ACTION_TYPE_ICON: Record<string, React.ReactNode> = {
  investigate_gap: (
    <svg className="h-4 w-4" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth={1.8}>
      <circle cx="11" cy="11" r="8" />
      <path strokeLinecap="round" strokeLinejoin="round" d="M21 21l-4.35-4.35" />
    </svg>
  ),
  bridge_exploration: (
    <svg className="h-4 w-4" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth={1.8}>
      <path
        strokeLinecap="round"
        strokeLinejoin="round"
        d="M13.828 10.172a4 4 0 00-5.656 0l-4 4a4 4 0 105.656 5.656l1.102-1.101m-.758-4.899a4 4 0 005.656 0l4-4a4 4 0 00-5.656-5.656l-1.1 1.1"
      />
    </svg>
  ),
};

const DefaultIcon = (
  <svg className="h-4 w-4" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth={1.8}>
    <path
      strokeLinecap="round"
      strokeLinejoin="round"
      d="M9 5H7a2 2 0 00-2 2v12a2 2 0 002 2h10a2 2 0 002-2V7a2 2 0 00-2-2h-2M9 5a2 2 0 002 2h2a2 2 0 002-2M9 5a2 2 0 012-2h2a2 2 0 012 2"
    />
  </svg>
);

type Props = {
  actions: PlanAction[];
  loading: boolean;
  graphId: string;
  nodes: GraphNode[];
};

function endpointNodeIds(action: PlanAction): [string | null, string | null] {
  const meta = (action.metadata ?? {}) as Record<string, unknown>;
  const a = typeof meta.node_a_id === "string" ? (meta.node_a_id as string) : null;
  const b = typeof meta.node_b_id === "string" ? (meta.node_b_id as string) : null;
  return [a, b];
}

function chunkIdsFromNode(node: GraphNode | undefined): string[] {
  if (!node) return [];
  const meta = (node.metadata ?? {}) as Record<string, unknown>;
  const ids = meta.source_chunk_ids;
  return Array.isArray(ids) ? (ids as string[]).filter(Boolean).slice(0, 1) : [];
}

export function GapPanel({ actions, loading, graphId, nodes }: Props) {
  const nodeMap = useMemo(() => new Map(nodes.map((n) => [n.id, n])), [nodes]);

  // Pull one representative chunk per endpoint node so each gap card can
  // show a "what we already know" excerpt from each side.
  const allChunkIds = useMemo(() => {
    const set = new Set<string>();
    actions.forEach((a) => {
      const [aId, bId] = endpointNodeIds(a);
      if (aId) chunkIdsFromNode(nodeMap.get(aId)).forEach((id) => set.add(id));
      if (bId) chunkIdsFromNode(nodeMap.get(bId)).forEach((id) => set.add(id));
    });
    return Array.from(set);
  }, [actions, nodeMap]);

  const [chunkMap, setChunkMap] = useState<Map<string, ChunkEvidence>>(new Map());

  useEffect(() => {
    let cancelled = false;
    if (allChunkIds.length === 0) {
      setChunkMap(new Map());
      return;
    }
    getChunkEvidence(graphId, allChunkIds)
      .then((rows) => {
        if (cancelled) return;
        const map = new Map<string, ChunkEvidence>();
        rows.forEach((r) => map.set(r.chunk_id, r));
        setChunkMap(map);
      })
      .catch(() => {
        if (!cancelled) setChunkMap(new Map());
      });
    return () => {
      cancelled = true;
    };
  }, [graphId, allChunkIds]);

  if (loading) {
    return (
      <div className="space-y-3 p-1">
        {[1, 2, 3].map((i) => (
          <div key={i} className="h-20 animate-pulse rounded-xl bg-white/4" />
        ))}
      </div>
    );
  }

  if (actions.length === 0) {
    return (
      <div className="rounded-xl border border-dashed border-white/10 py-10 text-center">
        <svg
          className="mx-auto mb-3 h-8 w-8 text-ink-3"
          viewBox="0 0 24 24"
          fill="none"
          stroke="currentColor"
          strokeWidth={1.5}
        >
          <path
            strokeLinecap="round"
            strokeLinejoin="round"
            d="M9 5H7a2 2 0 00-2 2v12a2 2 0 002 2h10a2 2 0 002-2V7a2 2 0 00-2-2h-2M9 5a2 2 0 002 2h2a2 2 0 002-2M9 5a2 2 0 012-2h2a2 2 0 012 2m-6 9l2 2 4-4"
          />
        </svg>
        <p className="text-sm text-ink-2">No research gaps identified</p>
        <p className="mt-1 text-xs text-ink-3">
          The gap analyst found no structural holes in this graph.
        </p>
      </div>
    );
  }

  return (
    <div className="space-y-3">
      {actions.map((action) => {
        const { label, style } = PRIORITY_LABEL(action.priority);
        const icon = ACTION_TYPE_ICON[action.action_type] ?? DefaultIcon;
        const [aId, bId] = endpointNodeIds(action);
        const aNode = aId ? nodeMap.get(aId) : undefined;
        const bNode = bId ? nodeMap.get(bId) : undefined;
        const aChunkId = chunkIdsFromNode(aNode)[0];
        const bChunkId = chunkIdsFromNode(bNode)[0];
        const aEv = aChunkId ? chunkMap.get(aChunkId) : undefined;
        const bEv = bChunkId ? chunkMap.get(bChunkId) : undefined;
        const aColor = aNode ? colorForDomain(String(aNode.cluster_id ?? "General Research")) : "#94a3b8";
        const bColor = bNode ? colorForDomain(String(bNode.cluster_id ?? "General Research")) : "#94a3b8";

        return (
          <div key={action.id} className="rounded-xl border border-white/8 bg-white/3 p-4">
            <div className="mb-2 flex items-start gap-3">
              <div className="mt-0.5 shrink-0 text-ink-2">{icon}</div>
              <div className="min-w-0 flex-1">
                <div className="mb-1 flex flex-wrap items-center gap-2">
                  <span
                    className={`rounded-full border px-2 py-0.5 text-[10px] font-semibold uppercase tracking-wide ${style}`}
                  >
                    {label} priority
                  </span>
                  <span className="text-[10px] capitalize text-ink-3">
                    {action.action_type.replace(/_/g, " ")}
                  </span>
                </div>
                <p className="text-xs font-medium text-ink">{action.description}</p>
              </div>
            </div>

            {action.rationale && (
              <p className="mt-2 pl-7 text-[11px] leading-relaxed text-ink-3">{action.rationale}</p>
            )}

            {(aNode || bNode) && (
              <div className="mt-3 grid gap-2 pl-7 sm:grid-cols-2">
                {aNode && (
                  <div>
                    <p
                      className="mb-1 inline-flex items-center gap-1.5 rounded-full border px-2 py-0.5 text-[10px] font-semibold"
                      style={{ borderColor: `${aColor}55`, color: aColor }}
                    >
                      <span className="h-1.5 w-1.5 rounded-full" style={{ background: aColor }} />
                      {aNode.label}
                    </p>
                    {aEv ? (
                      <EvidenceCard evidence={aEv} accent={aColor} compact />
                    ) : (
                      <p className="rounded-lg border border-dashed border-white/10 px-3 py-2 text-[10.5px] italic text-ink-3">
                        No excerpt available.
                      </p>
                    )}
                  </div>
                )}
                {bNode && (
                  <div>
                    <p
                      className="mb-1 inline-flex items-center gap-1.5 rounded-full border px-2 py-0.5 text-[10px] font-semibold"
                      style={{ borderColor: `${bColor}55`, color: bColor }}
                    >
                      <span className="h-1.5 w-1.5 rounded-full" style={{ background: bColor }} />
                      {bNode.label}
                    </p>
                    {bEv ? (
                      <EvidenceCard evidence={bEv} accent={bColor} compact />
                    ) : (
                      <p className="rounded-lg border border-dashed border-white/10 px-3 py-2 text-[10.5px] italic text-ink-3">
                        No excerpt available.
                      </p>
                    )}
                  </div>
                )}
              </div>
            )}
          </div>
        );
      })}
    </div>
  );
}
