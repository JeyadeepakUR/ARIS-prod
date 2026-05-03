"use client";

import { useEffect, useMemo, useState } from "react";

import { getChunkEvidence, type ChunkEvidence, type Contradiction } from "../../lib/api/graphs";
import { EvidenceCard } from "./EvidenceCard";

const SEVERITY_LABEL: Record<string, string> = {
  high: "High",
  medium: "Medium",
  low: "Low",
};

function severityBand(score: number): "high" | "medium" | "low" {
  if (score >= 0.7) return "high";
  if (score >= 0.4) return "medium";
  return "low";
}

const SEVERITY_STYLES = {
  high: "border-red-500/30 bg-red-500/8 text-red-300",
  medium: "border-amber-500/30 bg-amber-500/8 text-amber-300",
  low: "border-white/10 bg-white/4 text-ink-2",
};

const TYPE_LABELS: Record<string, string> = {
  direct: "Direct conflict",
  methodological: "Methodological",
  interpretive: "Interpretive",
  scope: "Scope mismatch",
};

type Props = {
  contradictions: Contradiction[];
  loading: boolean;
  graphId: string;
};

export function ContradictionsPanel({ contradictions, loading, graphId }: Props) {
  // Pre-fetch evidence for ALL contradictions in a single round-trip so each
  // card can render the literal source paragraphs without N round-trips.
  const allChunkIds = useMemo(() => {
    const ids = new Set<string>();
    contradictions.forEach((c) => {
      if (c.claim_a_chunk_id) ids.add(c.claim_a_chunk_id);
      if (c.claim_b_chunk_id) ids.add(c.claim_b_chunk_id);
    });
    return Array.from(ids);
  }, [contradictions]);

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
        {[1, 2].map((i) => (
          <div key={i} className="h-24 animate-pulse rounded-xl bg-white/4" />
        ))}
      </div>
    );
  }

  if (contradictions.length === 0) {
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
            d="M9 12l2 2 4-4m5.618-4.016A11.955 11.955 0 0112 2.944a11.955 11.955 0 01-8.618 3.04A12.02 12.02 0 003 9c0 5.591 3.824 10.29 9 11.622 5.176-1.332 9-6.03 9-11.622 0-1.042-.133-2.052-.382-3.016z"
          />
        </svg>
        <p className="text-sm text-ink-2">No contradictions detected</p>
        <p className="mt-1 text-xs text-ink-3">
          The agent found no conflicting claims across documents.
        </p>
      </div>
    );
  }

  return (
    <div className="space-y-3">
      {contradictions.map((c) => {
        const band = severityBand(c.severity);
        const typeLabel = TYPE_LABELS[c.contradiction_type] ?? c.contradiction_type;
        const evidenceA = c.claim_a_chunk_id ? chunkMap.get(c.claim_a_chunk_id) : undefined;
        const evidenceB = c.claim_b_chunk_id ? chunkMap.get(c.claim_b_chunk_id) : undefined;

        return (
          <div key={c.id} className={`rounded-xl border p-4 ${SEVERITY_STYLES[band]}`}>
            <div className="mb-3 flex flex-wrap items-center gap-2">
              <span
                className={`rounded-full border px-2 py-0.5 text-[10px] font-semibold uppercase tracking-wide ${SEVERITY_STYLES[band]}`}
              >
                {SEVERITY_LABEL[band]} severity
              </span>
              <span className="rounded-full border border-white/10 bg-white/5 px-2 py-0.5 text-[10px] text-ink-2">
                {typeLabel}
              </span>
              <span className="ml-auto text-[10px] text-ink-3">
                {new Date(c.created_at).toLocaleDateString("en-US", {
                  month: "short",
                  day: "numeric",
                })}
              </span>
            </div>

            <div className="grid gap-3 sm:grid-cols-2">
              <div>
                <p className="mb-1.5 flex items-center gap-1.5 text-[10px] font-semibold uppercase tracking-wider text-ink-3">
                  <span className="h-1.5 w-1.5 rounded-full bg-rose-400" />
                  Claim A {c.author_a ? `· ${c.author_a}` : ""}
                </p>
                <p className="mb-2 rounded-lg border border-rose-500/20 bg-rose-500/5 px-3 py-2 text-[12px] font-medium leading-relaxed text-ink">
                  {c.claim_a_text}
                </p>
                {evidenceA ? (
                  <EvidenceCard evidence={evidenceA} accent="#fb7185" compact />
                ) : (
                  <p className="rounded-lg border border-dashed border-white/10 px-3 py-2 text-[10.5px] italic text-ink-3">
                    No source excerpt stored.
                  </p>
                )}
              </div>
              <div>
                <p className="mb-1.5 flex items-center gap-1.5 text-[10px] font-semibold uppercase tracking-wider text-ink-3">
                  <span className="h-1.5 w-1.5 rounded-full bg-sky-400" />
                  Claim B {c.author_b ? `· ${c.author_b}` : ""}
                </p>
                <p className="mb-2 rounded-lg border border-sky-500/20 bg-sky-500/5 px-3 py-2 text-[12px] font-medium leading-relaxed text-ink">
                  {c.claim_b_text}
                </p>
                {evidenceB ? (
                  <EvidenceCard evidence={evidenceB} accent="#38bdf8" compact />
                ) : (
                  <p className="rounded-lg border border-dashed border-white/10 px-3 py-2 text-[10.5px] italic text-ink-3">
                    No source excerpt stored.
                  </p>
                )}
              </div>
            </div>

            {c.llm_reasoning && (
              <div className="mt-3 rounded-lg border border-white/8 bg-white/3 px-3 py-2">
                <p className="mb-0.5 text-[10px] font-semibold uppercase tracking-wider text-ink-3">
                  Why this is a contradiction
                </p>
                <p className="text-[11px] italic leading-relaxed text-ink-2">{c.llm_reasoning}</p>
              </div>
            )}
          </div>
        );
      })}
    </div>
  );
}
