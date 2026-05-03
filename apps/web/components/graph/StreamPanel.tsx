"use client";

import type { StreamEvent, StreamStatus } from "../../lib/hooks/useGraphStream";

const AGENT_LABELS: Record<string, string> = {
  orchestrator: "Orchestrator",
  concept_extractor: "Concept Extractor",
  bridge_discoverer: "Bridge Discoverer",
  contradiction_analyst: "Contradiction Analyst",
  hypothesis_formulator: "Hypothesis Formulator",
  gap_analyst: "Gap Analyst",
  system: "System",
};

const AGENT_COLORS: Record<string, string> = {
  orchestrator: "text-violet-300",
  concept_extractor: "text-sky-300",
  bridge_discoverer: "text-amber-300",
  contradiction_analyst: "text-rose-300",
  hypothesis_formulator: "text-emerald-300",
  gap_analyst: "text-orange-300",
  system: "text-ink-2",
};

function EventRow({ evt }: { evt: StreamEvent }) {
  const node = evt.agent_node ?? "system";
  const label = AGENT_LABELS[node] ?? node;
  const color = AGENT_COLORS[node] ?? "text-ink-2";
  const isTerminal = evt.event_type === "run_complete";
  const isError = evt.event_type === "error";

  if (isTerminal) {
    return (
      <div className="flex items-center gap-2 py-1">
        <svg className="h-4 w-4 shrink-0 text-emerald-400" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth={2.5}>
          <path strokeLinecap="round" strokeLinejoin="round" d="M9 12l2 2 4-4m6 2a9 9 0 11-18 0 9 9 0 0118 0z"/>
        </svg>
        <span className="text-xs font-semibold text-emerald-300">Analysis complete</span>
      </div>
    );
  }

  if (isError) {
    const msg = (evt.content?.error as string) ?? "Unknown error";
    return (
      <div className="flex items-start gap-2 py-1">
        <svg className="mt-0.5 h-4 w-4 shrink-0 text-red-400" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth={2}>
          <path strokeLinecap="round" strokeLinejoin="round" d="M12 9v4m0 4h.01M10.29 3.86L1.82 18a2 2 0 001.71 3h16.94a2 2 0 001.71-3L13.71 3.86a2 2 0 00-3.42 0z"/>
        </svg>
        <span className="text-xs text-red-300">{msg}</span>
      </div>
    );
  }

  const summary = evt.content
    ? Object.entries(evt.content)
        .filter(([k]) => k !== "error")
        .map(([k, v]) => `${k}: ${v}`)
        .join(" · ")
    : null;

  return (
    <div className="flex items-start gap-2 py-1 border-b border-white/5 last:border-0">
      <div className="mt-1 h-1.5 w-1.5 shrink-0 rounded-full bg-white/20" />
      <div className="min-w-0">
        <span className={`text-[11px] font-semibold ${color}`}>{label}</span>
        {" · "}
        <span className="text-[11px] text-ink-2">{evt.event_type}</span>
        {summary && (
          <p className="mt-0.5 text-[11px] text-ink-3 truncate">{summary}</p>
        )}
      </div>
    </div>
  );
}

type Props = {
  events: StreamEvent[];
  status: StreamStatus;
  error: string | null;
};

export function StreamPanel({ events, status, error }: Props) {
  const isLive = status === "connecting" || status === "streaming";

  return (
    <div className="mt-4 glass rounded-2xl p-4">
      <div className="mb-3 flex items-center gap-2">
        {isLive && (
          <span className="relative flex h-2 w-2">
            <span className="absolute inline-flex h-full w-full animate-ping rounded-full bg-accent opacity-75" />
            <span className="relative inline-flex h-2 w-2 rounded-full bg-accent" />
          </span>
        )}
        <p className="text-[11px] font-semibold uppercase tracking-[0.18em] text-ink-2">
          {status === "connecting" ? "Connecting…" :
           status === "streaming" ? "Analyzing…" :
           status === "complete" ? "Agent trace" :
           status === "error" ? "Stream error" : "Agent trace"}
        </p>
        <span className="ml-auto text-[11px] text-ink-3">{events.length} events</span>
      </div>

      {error && (
        <p className="mb-3 rounded-lg border border-red-500/20 bg-red-500/8 px-3 py-2 text-xs text-red-300">{error}</p>
      )}

      {events.length === 0 && status === "connecting" ? (
        <div className="space-y-2">
          {[1, 2, 3].map((i) => (
            <div key={i} className="h-5 animate-pulse rounded bg-white/5" />
          ))}
        </div>
      ) : (
        <div className="max-h-48 overflow-y-auto space-y-0 pr-1">
          {events.map((evt, i) => (
            <EventRow key={evt.id ?? i} evt={evt} />
          ))}
        </div>
      )}
    </div>
  );
}
