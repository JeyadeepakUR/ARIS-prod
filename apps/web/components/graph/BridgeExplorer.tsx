"use client";

import { useMemo, useState } from "react";

import type { GraphEdge, GraphNode, Hypothesis, HypothesisStatus } from "../../lib/api/graphs";

const STATUS_OPTIONS: HypothesisStatus[] = [
  "proposed",
  "investigating",
  "accepted",
  "rejected",
];

const STATUS_STYLES: Record<HypothesisStatus, string> = {
  proposed: "border-white/12 bg-white/6 text-ink-2",
  investigating: "border-amber-500/30 bg-amber-500/10 text-amber-300",
  accepted: "border-emerald-500/30 bg-emerald-500/10 text-emerald-300",
  rejected: "border-red-500/30 bg-red-500/10 text-red-300",
};

function domainPairKey(edge: GraphEdge, nodes: GraphNode[]): string {
  const src = nodes.find((n) => n.id === edge.source_node_id);
  const tgt = nodes.find((n) => n.id === edge.target_node_id);
  const a = src?.cluster_id ?? "Unknown";
  const b = tgt?.cluster_id ?? "Unknown";
  return [a, b].sort().join(" ↔ ");
}

type BridgeCardProps = {
  edge: GraphEdge;
  nodes: GraphNode[];
  hypotheses: Hypothesis[];
  onUpdateStatus: (hypothesisId: string, status: HypothesisStatus) => Promise<void>;
  hypothesesLoading: boolean;
};

function BridgeCard({ edge, nodes, hypotheses, onUpdateStatus, hypothesesLoading }: BridgeCardProps) {
  const [expanded, setExpanded] = useState(false);
  const [updating, setUpdating] = useState<string | null>(null);

  const src = nodes.find((n) => n.id === edge.source_node_id);
  const tgt = nodes.find((n) => n.id === edge.target_node_id);
  const srcLabel = src?.label ?? "Source";
  const tgtLabel = tgt?.label ?? "Target";
  const srcDomain = src?.cluster_id ?? "Unknown";
  const tgtDomain = tgt?.cluster_id ?? "Unknown";
  const bridgeConcept = edge.bridge_concept ?? "Cross-domain link";
  const pct = Math.round(edge.confidence * 100);
  const edgeHypotheses = hypotheses.filter((h) => h.edge_id === edge.id);

  async function handleStatus(hId: string, status: HypothesisStatus) {
    setUpdating(hId);
    try {
      await onUpdateStatus(hId, status);
    } finally {
      setUpdating(null);
    }
  }

  return (
    <div className="rounded-2xl border border-white/8 bg-white/3 overflow-hidden">
      {/* Bridge header */}
      <div className="flex flex-wrap items-start gap-3 p-4">
        <span className="shrink-0 inline-flex rounded-full border border-orange-500/30 bg-orange-500/10 px-3 py-1 text-xs font-semibold text-orange-300">
          {bridgeConcept}
        </span>

        <div className="flex flex-wrap items-center gap-1.5 text-xs min-w-0">
          <span className="rounded-lg border border-white/10 bg-white/5 px-2 py-1 font-medium text-ink truncate max-w-[140px]" title={srcLabel}>
            {srcLabel}
          </span>
          <svg className="h-3 w-3 shrink-0 text-ink-3" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}>
            <path strokeLinecap="round" strokeLinejoin="round" d="M14 5l7 7m0 0l-7 7m7-7H3"/>
          </svg>
          <span className="rounded-lg border border-white/10 bg-white/5 px-2 py-1 font-medium text-ink truncate max-w-[140px]" title={tgtLabel}>
            {tgtLabel}
          </span>
        </div>

        <div className="ml-auto flex items-center gap-2 shrink-0">
          <span className="text-[11px] text-ink-3">{srcDomain} ↔ {tgtDomain}</span>
          <span className={`rounded-full border px-2 py-0.5 text-[10px] font-bold ${
            pct >= 70 ? "border-emerald-500/30 bg-emerald-500/10 text-emerald-300" :
            pct >= 40 ? "border-amber-500/30 bg-amber-500/10 text-amber-300" :
            "border-white/10 bg-white/5 text-ink-2"
          }`}>
            {pct}%
          </span>
        </div>
      </div>

      {/* Confidence bar */}
      <div className="px-4 pb-1">
        <div className="h-1 w-full rounded-full bg-white/8">
          <div
            className="h-1 rounded-full bg-gradient-to-r from-accent to-bridge transition-all"
            style={{ width: `${pct}%` }}
          />
        </div>
      </div>

      {/* Hypotheses */}
      <div className="px-4 pt-3 pb-4 space-y-3">
        {hypothesesLoading ? (
          <div className="space-y-2">
            <div className="h-4 animate-pulse rounded bg-white/5 w-3/4" />
            <div className="h-4 animate-pulse rounded bg-white/5 w-1/2" />
          </div>
        ) : edgeHypotheses.length === 0 ? (
          <p className="text-xs text-ink-3 italic">No hypotheses generated for this bridge.</p>
        ) : (
          edgeHypotheses.map((h) => (
            <div key={h.id} className="rounded-xl border border-accent/15 bg-accent/5 p-3">
              <p className="text-xs leading-relaxed text-ink-2 mb-3">{h.hypothesis_text}</p>
              <div className="flex flex-wrap gap-1.5">
                {STATUS_OPTIONS.map((s) => {
                  const isActive = h.status === s;
                  return (
                    <button
                      key={s}
                      type="button"
                      disabled={updating === h.id}
                      onClick={() => handleStatus(h.id, s)}
                      className={`rounded-full border px-2.5 py-0.5 text-[10px] font-semibold transition disabled:opacity-50 ${
                        isActive
                          ? STATUS_STYLES[s]
                          : "border-white/8 bg-white/3 text-ink-3 hover:border-white/14 hover:text-ink-2"
                      }`}
                    >
                      {s}
                    </button>
                  );
                })}
              </div>
            </div>
          ))
        )}

        {/* Evidence collapsible */}
        {edge.evidence?.text && (
          <button
            type="button"
            onClick={() => setExpanded((v) => !v)}
            className="flex w-full items-center gap-2 text-left"
          >
            <svg
              className={`h-3 w-3 shrink-0 text-ink-3 transition-transform ${expanded ? "rotate-90" : ""}`}
              fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}
            >
              <path strokeLinecap="round" strokeLinejoin="round" d="M9 5l7 7-7 7"/>
            </svg>
            <span className="text-[11px] text-ink-3 hover:text-ink-2 transition">Evidence</span>
          </button>
        )}
        {expanded && edge.evidence?.text && (
          <p className="ml-5 text-[11px] text-ink-2 leading-relaxed border-l border-white/10 pl-3">
            {edge.evidence.text}
          </p>
        )}
      </div>
    </div>
  );
}

type Props = {
  bridges: GraphEdge[];
  nodes: GraphNode[];
  hypotheses: Hypothesis[];
  hypothesesLoading: boolean;
  onUpdateStatus: (hypothesisId: string, status: HypothesisStatus) => Promise<void>;
};

export function BridgeExplorer({ bridges, nodes, hypotheses, hypothesesLoading, onUpdateStatus }: Props) {
  const [domainFilter, setDomainFilter] = useState<string>("all");
  const [minConfidence, setMinConfidence] = useState(0);
  const [sortBy, setSortBy] = useState<"confidence" | "concept">("confidence");

  const domainPairs = useMemo(() => {
    const pairs = new Set<string>();
    bridges.forEach((e) => pairs.add(domainPairKey(e, nodes)));
    return ["all", ...Array.from(pairs).sort()];
  }, [bridges, nodes]);

  const filtered = useMemo(() => {
    let result = bridges.filter((e) => e.confidence >= minConfidence);
    if (domainFilter !== "all") {
      result = result.filter((e) => domainPairKey(e, nodes) === domainFilter);
    }
    if (sortBy === "confidence") {
      result = [...result].sort((a, b) => b.confidence - a.confidence);
    } else {
      result = [...result].sort((a, b) =>
        (a.bridge_concept ?? "").localeCompare(b.bridge_concept ?? ""),
      );
    }
    return result;
  }, [bridges, nodes, domainFilter, minConfidence, sortBy]);

  if (bridges.length === 0) {
    return (
      <div className="rounded-xl border border-dashed border-white/10 py-12 text-center">
        <svg className="mx-auto mb-3 h-10 w-10 text-ink-3" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth={1.5}>
          <path strokeLinecap="round" strokeLinejoin="round" d="M13.828 10.172a4 4 0 00-5.656 0l-4 4a4 4 0 105.656 5.656l1.102-1.101m-.758-4.899a4 4 0 005.656 0l4-4a4 4 0 00-5.656-5.656l-1.1 1.1"/>
        </svg>
        <p className="text-sm text-ink-2">No cross-domain bridges found</p>
        <p className="mt-1 text-xs text-ink-3">Bridges appear when concepts from different domains share high semantic similarity.</p>
      </div>
    );
  }

  return (
    <div>
      {/* Filters */}
      <div className="mb-5 flex flex-wrap items-center gap-3">
        {/* Domain pair filter */}
        <div className="flex flex-wrap gap-1.5">
          {domainPairs.slice(0, 6).map((pair) => (
            <button
              key={pair}
              type="button"
              onClick={() => setDomainFilter(pair)}
              className={`rounded-full border px-2.5 py-1 text-[11px] font-semibold transition ${
                domainFilter === pair
                  ? "border-accent/40 bg-accent/15 text-accent-2"
                  : "border-white/10 bg-white/4 text-ink-3 hover:border-white/16 hover:text-ink-2"
              }`}
            >
              {pair === "all" ? `All bridges (${bridges.length})` : pair}
            </button>
          ))}
        </div>

        <div className="ml-auto flex items-center gap-3">
          {/* Min confidence */}
          <label className="flex items-center gap-2 text-[11px] text-ink-3">
            Min confidence
            <input
              type="range"
              min={0}
              max={1}
              step={0.05}
              value={minConfidence}
              onChange={(e) => setMinConfidence(Number(e.target.value))}
              className="w-24 accent-accent"
            />
            <span className="w-8 text-right text-ink-2">{Math.round(minConfidence * 100)}%</span>
          </label>

          {/* Sort */}
          <select
            value={sortBy}
            onChange={(e) => setSortBy(e.target.value as "confidence" | "concept")}
            className="rounded-lg border border-white/10 bg-bg-2 px-2 py-1 text-[11px] text-ink-2 outline-none"
          >
            <option value="confidence">Sort: confidence</option>
            <option value="concept">Sort: concept</option>
          </select>
        </div>
      </div>

      {/* Results count */}
      <p className="mb-3 text-[11px] text-ink-3">
        {filtered.length} bridge{filtered.length !== 1 ? "s" : ""}
        {filtered.length !== bridges.length ? ` of ${bridges.length}` : ""}
      </p>

      {filtered.length === 0 ? (
        <div className="rounded-xl border border-dashed border-white/10 py-8 text-center">
          <p className="text-sm text-ink-2">No bridges match these filters.</p>
        </div>
      ) : (
        <div className="space-y-3">
          {filtered.map((edge) => (
            <BridgeCard
              key={edge.id}
              edge={edge}
              nodes={nodes}
              hypotheses={hypotheses}
              hypothesesLoading={hypothesesLoading}
              onUpdateStatus={onUpdateStatus}
            />
          ))}
        </div>
      )}
    </div>
  );
}
