"use client";

import { useState } from "react";

import { buildDocumentDownloadUrl, type ChunkEvidence } from "../../lib/api/graphs";

type Props = {
  evidence: ChunkEvidence;
  accent?: string;
  compact?: boolean;
};

const MAX_PREVIEW_CHARS = 320;

export function EvidenceCard({ evidence, accent = "#94a3b8", compact = false }: Props) {
  const [expanded, setExpanded] = useState(false);
  const text = (evidence.content ?? "").trim();
  const truncated = text.length > MAX_PREVIEW_CHARS;
  const preview = truncated && !expanded ? `${text.slice(0, MAX_PREVIEW_CHARS).trim()}…` : text;

  const pdfUrl = evidence.document_s3_key
    ? buildDocumentDownloadUrl(evidence.document_s3_key, evidence.page_number)
    : null;

  return (
    <article
      className="rounded-xl border bg-white/3 px-3 py-2.5 transition hover:bg-white/5"
      style={{ borderColor: `${accent}40` }}
    >
      <header className="flex items-start justify-between gap-2">
        <div className="min-w-0">
          {pdfUrl ? (
            <a
              href={pdfUrl}
              target="_blank"
              rel="noopener noreferrer"
              className="block truncate text-[12px] font-semibold text-ink hover:underline"
              title={evidence.document_filename}
              style={{ color: accent }}
            >
              {evidence.document_title}
            </a>
          ) : (
            <p className="truncate text-[12px] font-semibold text-ink" title={evidence.document_filename}>
              {evidence.document_title}
            </p>
          )}
          <p className="mt-0.5 flex flex-wrap items-center gap-1.5 text-[10px] uppercase tracking-wider text-ink-3">
            {evidence.page_number !== null && evidence.page_number !== undefined && (
              <span className="rounded bg-white/8 px-1.5 py-0.5 font-mono">p.{evidence.page_number}</span>
            )}
            {evidence.section && (
              <span className="truncate" title={evidence.section}>
                {evidence.section}
              </span>
            )}
            {pdfUrl && (
              <a
                href={pdfUrl}
                target="_blank"
                rel="noopener noreferrer"
                className="ml-auto inline-flex items-center gap-0.5 text-accent hover:underline"
              >
                Open PDF
                <svg className="h-2.5 w-2.5" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth={2}>
                  <path strokeLinecap="round" strokeLinejoin="round" d="M14 3h7v7M21 3l-9 9M5 5h6v2H7v10h10v-4h2v6H5z" />
                </svg>
              </a>
            )}
          </p>
        </div>
      </header>
      <blockquote
        className={`mt-2 border-l-2 pl-2 text-[11.5px] italic leading-relaxed text-ink-2 ${compact ? "" : "whitespace-pre-wrap"}`}
        style={{ borderColor: accent }}
      >
        {preview || <span className="opacity-50">(empty chunk)</span>}
      </blockquote>
      {truncated && (
        <button
          type="button"
          onClick={() => setExpanded((v) => !v)}
          className="mt-1.5 text-[10px] font-semibold text-ink-3 hover:text-ink"
        >
          {expanded ? "Show less" : `Show full chunk (${text.length} chars)`}
        </button>
      )}
    </article>
  );
}

type StackProps = {
  evidence: ChunkEvidence[];
  accent?: string;
  emptyText?: string;
};

export function EvidenceStack({ evidence, accent, emptyText }: StackProps) {
  if (!evidence.length) {
    return (
      <p className="rounded-lg border border-dashed border-white/10 px-3 py-2 text-[11px] italic text-ink-3">
        {emptyText ?? "No source excerpts available."}
      </p>
    );
  }
  return (
    <ul className="space-y-2">
      {evidence.map((ev) => (
        <li key={ev.chunk_id}>
          <EvidenceCard evidence={ev} accent={accent} />
        </li>
      ))}
    </ul>
  );
}
