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
  node: GraphNode | null;
  graphId: string;
  allNodes: GraphNode[];
  allEdges: GraphEdge[];
  onClose: () => void;
  onFocusNode?: (nodeId: string) => void;
};

export function NodeInspector({ node, graphId, allNodes, allEdges, onClose, onFocusNode }: Props) {
  const detail = useMemo(() => {
    if (!node) return null;
    const nodeMap = new Map(allNodes.map((n) => [n.id, n]));

    const incoming = allEdges.filter((e) => e.target_node_id === node.id);
    const outgoing = allEdges.filter((e) => e.source_node_id === node.id);

    const sourceDocuments = outgoing
      .filter((e) => e.edge_type === "extracted_from")
      .map((e) => nodeMap.get(e.target_node_id))
      .filter((n): n is GraphNode => Boolean(n));

    const concepts = outgoing
      .filter((e) => e.edge_type === "has_concept")
      .map((e) => nodeMap.get(e.target_node_id))
      .filter((n): n is GraphNode => Boolean(n));

    const bridges = [...incoming, ...outgoing].filter(
      (e) => e.edge_type === "cross_domain_bridge",
    );

    const containingDomain = incoming.find((e) => e.edge_type === "has_concept");
    const domainNode = containingDomain ? nodeMap.get(containingDomain.source_node_id) : null;

    return { sourceDocuments, concepts, bridges, domainNode };
  }, [node, allNodes, allEdges]);

  // Fetch evidence chunks the LLM actually read for this concept.
  const sourceChunkIds = useMemo<string[]>(() => {
    if (!node) return [];
    const meta = (node.metadata ?? {}) as Record<string, unknown>;
    const ids = Array.isArray(meta.source_chunk_ids) ? (meta.source_chunk_ids as string[]) : [];
    return ids.filter(Boolean);
  }, [node]);

  const [evidence, setEvidence] = useState<ChunkEvidence[]>([]);
  const [evidenceLoading, setEvidenceLoading] = useState(false);
  const [evidenceError, setEvidenceError] = useState<string | null>(null);

  useEffect(() => {
    let cancelled = false;
    setEvidence([]);
    setEvidenceError(null);
    if (!node || sourceChunkIds.length === 0) return;
    setEvidenceLoading(true);
    getChunkEvidence(graphId, sourceChunkIds)
      .then((rows) => {
        if (!cancelled) setEvidence(rows);
      })
      .catch((err) => {
        if (!cancelled) setEvidenceError(err instanceof Error ? err.message : "Failed to load evidence");
      })
      .finally(() => {
        if (!cancelled) setEvidenceLoading(false);
      });
    return () => {
      cancelled = true;
    };
  }, [node, graphId, sourceChunkIds]);

  // Group evidence by source paper for cleaner display.
  const evidenceByDoc = useMemo(() => {
    const groups = new Map<string, { title: string; items: ChunkEvidence[] }>();
    evidence.forEach((ev) => {
      const key = ev.document_id;
      const existing = groups.get(key);
      if (existing) {
        existing.items.push(ev);
      } else {
        groups.set(key, { title: ev.document_title, items: [ev] });
      }
    });
    return Array.from(groups.entries());
  }, [evidence]);

  if (!node || !detail) return null;

  const cluster = String(node.cluster_id ?? "General Research");
  const color = colorForDomain(cluster);
  const meta = (node.metadata ?? {}) as Record<string, unknown>;
  const confidence = typeof meta?.confidence === "number" ? (meta.confidence as number) : null;
  const isConcept = node.node_type === "concept";
  const isDomain = node.node_type === "domain";
  const isDocument = node.node_type === "document";

  return (
    <aside
      className="fixed right-4 top-24 z-30 flex w-[420px] max-h-[82vh] flex-col rounded-2xl border border-white/10 bg-[#11131a]/95 shadow-2xl backdrop-blur"
      style={{ boxShadow: `0 8px 30px ${color}33` }}
    >
      <div
        className="flex items-start justify-between gap-3 border-b border-white/10 p-4"
        style={{ background: `${color}15` }}
      >
        <div className="min-w-0">
          <p className="text-[10px] font-semibold uppercase tracking-wider opacity-70">
            {node.node_type.replace("_", " ")}
          </p>
          <h3 className="mt-1 text-sm font-bold leading-tight text-ink">{node.label}</h3>
          {isConcept && (
            <p
              className="mt-1.5 inline-flex items-center gap-1.5 rounded-full px-2 py-0.5 text-[10px] font-semibold"
              style={{ background: `${color}25`, color }}
            >
              <span className="h-1.5 w-1.5 rounded-full" style={{ background: color }} />
              {cluster}
            </p>
          )}
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
      </div>

      <div className="flex-1 space-y-4 overflow-y-auto p-4 text-xs text-ink-2">
        {confidence !== null && (
          <div className="flex items-center gap-2">
            <span className="opacity-70">Confidence:</span>
            <span className="font-mono font-semibold text-ink">{(confidence * 100).toFixed(0)}%</span>
            <div className="h-1.5 flex-1 overflow-hidden rounded-full bg-white/5">
              <div
                className="h-full rounded-full"
                style={{ width: `${confidence * 100}%`, background: color }}
              />
            </div>
          </div>
        )}

        {isConcept && (
          <Section
            label={`Source excerpts${evidence.length ? ` (${evidence.length})` : ""}`}
            hint="Literal text the LLM read to extract this concept"
          >
            {evidenceLoading && (
              <p className="text-[10px] italic text-ink-3">Loading evidence…</p>
            )}
            {evidenceError && (
              <p className="text-[10px] italic text-red-300">{evidenceError}</p>
            )}
            {!evidenceLoading && !evidenceError && evidence.length === 0 && (
              <p className="rounded-lg border border-dashed border-white/10 px-3 py-2 text-[10.5px] italic text-ink-3">
                No source chunk references stored for this concept.
              </p>
            )}
            {evidenceByDoc.length > 0 && (
              <div className="space-y-3">
                {evidenceByDoc.map(([docId, group]) => (
                  <div key={docId}>
                    {evidenceByDoc.length > 1 && (
                      <p className="mb-1.5 text-[10px] font-semibold uppercase tracking-wider text-ink-3">
                        {group.title}
                      </p>
                    )}
                    <EvidenceStack evidence={group.items} accent={color} />
                  </div>
                ))}
              </div>
            )}
          </Section>
        )}

        {detail.domainNode && isConcept && (
          <Section label="Domain">
            <button
              type="button"
              onClick={() => onFocusNode?.(detail.domainNode!.id)}
              className="rounded-full border px-2 py-0.5 text-[10px] font-medium transition hover:bg-white/8"
              style={{ borderColor: `${color}55`, color: "#e5e7eb" }}
            >
              <span className="mr-1 inline-block h-1.5 w-1.5 rounded-full align-middle" style={{ background: color }} />
              {detail.domainNode.label}
            </button>
          </Section>
        )}

        {detail.bridges.length > 0 && (
          <Section label={`Cross-domain bridges (${detail.bridges.length})`}>
            <ul className="space-y-1.5">
              {detail.bridges.slice(0, 8).map((b) => (
                <li
                  key={b.id}
                  className="rounded-lg border border-amber-500/30 bg-amber-500/10 px-2.5 py-1.5"
                >
                  <p className="truncate text-[11px] font-semibold text-amber-200">
                    {b.bridge_concept ?? "Bridge"}
                  </p>
                  <p className="mt-0.5 text-[10px] text-amber-100/80">
                    confidence {(b.confidence * 100).toFixed(0)}%
                  </p>
                </li>
              ))}
            </ul>
          </Section>
        )}

        {isDomain && detail.concepts.length > 0 && (
          <Section label={`Concepts in this domain (${detail.concepts.length})`}>
            <div className="flex flex-wrap gap-1.5">
              {detail.concepts.slice(0, 40).map((c) => (
                <button
                  type="button"
                  key={c.id}
                  onClick={() => onFocusNode?.(c.id)}
                  className="rounded-full border px-2 py-0.5 text-[10px] font-medium transition hover:bg-white/8"
                  style={{ borderColor: `${color}55`, color: "#e5e7eb" }}
                >
                  {c.label}
                </button>
              ))}
              {detail.concepts.length > 40 && (
                <span className="text-[10px] opacity-60">+{detail.concepts.length - 40} more</span>
              )}
            </div>
          </Section>
        )}

        {isDocument && (
          <Section label="Document" hint="This node represents a source paper">
            <p className="text-[11px] leading-relaxed text-ink-2">
              Click any concept extracted from this paper to see the literal sentences from the
              text that produced it.
            </p>
          </Section>
        )}

        {detail.sourceDocuments.length > 0 && (
          <Section label={`Source paper${detail.sourceDocuments.length === 1 ? "" : "s"}`}>
            <ul className="space-y-1">
              {detail.sourceDocuments.map((doc) => (
                <li
                  key={doc.id}
                  className="flex items-center gap-2 rounded-lg border border-white/8 bg-white/3 px-2 py-1.5"
                >
                  <svg
                    className="h-3.5 w-3.5 flex-shrink-0 opacity-60"
                    viewBox="0 0 24 24"
                    fill="none"
                    stroke="currentColor"
                    strokeWidth={2}
                  >
                    <path
                      strokeLinecap="round"
                      strokeLinejoin="round"
                      d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z"
                    />
                    <path strokeLinecap="round" strokeLinejoin="round" d="M14 2v6h6" />
                  </svg>
                  <button
                    type="button"
                    onClick={() => onFocusNode?.(doc.id)}
                    className="truncate text-[11px] hover:underline"
                  >
                    {doc.label}
                  </button>
                </li>
              ))}
            </ul>
          </Section>
        )}
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
