"use client";

import Link from "next/link";
import { useParams } from "next/navigation";
import { useCallback, useEffect, useMemo, useState } from "react";

import { PageHeader } from "../../../../../components/layout/PageHeader";
import { JobStatusBadge } from "../../../../../components/shared/JobStatusBadge";
import { listDocuments, type DocumentItem } from "../../../../../lib/api/documents";
import { buildGraph, listGraphs, type Graph } from "../../../../../lib/api/graphs";
import { useJobPoller } from "../../../../../lib/hooks/useJobPoller";

export default function WorkspaceGraphsPage() {
  const params = useParams<{ id: string }>();
  const workspaceId = params.id;

  const [documents, setDocuments] = useState<DocumentItem[]>([]);
  const [graphs, setGraphs] = useState<Graph[]>([]);
  const [selectedDocumentIds, setSelectedDocumentIds] = useState<string[]>([]);
  const [activeJobId, setActiveJobId] = useState<string | null>(null);
  const [loading, setLoading] = useState(true);
  const [submitting, setSubmitting] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const { job, error: jobError, isComplete } = useJobPoller(activeJobId, Boolean(activeJobId));

  const loadData = useCallback(async () => {
    if (!workspaceId) {
      return;
    }
    const [docs, graphRows] = await Promise.all([listDocuments(workspaceId), listGraphs(workspaceId)]);
    setDocuments(docs);
    setGraphs(graphRows);
  }, [workspaceId]);

  useEffect(() => {
    let mounted = true;
    setLoading(true);
    loadData()
      .catch((loadError) => {
        if (mounted) {
          setError(loadError instanceof Error ? loadError.message : "Failed to load graph resources");
        }
      })
      .finally(() => {
        if (mounted) {
          setLoading(false);
        }
      });

    return () => {
      mounted = false;
    };
  }, [loadData]);

  useEffect(() => {
    if (isComplete) {
      loadData().catch(() => undefined);
    }
  }, [isComplete, loadData]);

  const readyDocuments = useMemo(() => documents.filter((document) => document.status === "ready"), [documents]);

  function toggleDocument(documentId: string) {
    setSelectedDocumentIds((current) =>
      current.includes(documentId)
        ? current.filter((id) => id !== documentId)
        : [...current, documentId],
    );
  }

  async function triggerBuild() {
    if (!workspaceId || selectedDocumentIds.length < 2) {
      return;
    }

    setSubmitting(true);
    setError(null);
    try {
      const response = await buildGraph(workspaceId, selectedDocumentIds);
      setActiveJobId(response.job_id);
    } catch (buildError) {
      setError(buildError instanceof Error ? buildError.message : "Failed to start graph build");
    } finally {
      setSubmitting(false);
    }
  }

  return (
    <div>
      <PageHeader
        eyebrow="Graph Builder"
        title="Knowledge Graphs"
        description="Select 2+ documents and build a domain concept network with cross-domain bridge detection."
        action={
          <Link href={`/workspaces/${workspaceId}/documents`}
            className="rounded-xl border border-white/10 bg-white/5 px-4 py-2 text-xs font-semibold text-ink-2 transition hover:bg-white/10 hover:text-ink">
            ← Documents
          </Link>
        }
      />

      {/* Build panel */}
      <div className="mb-6 glass rounded-2xl p-5">
        <p className="mb-4 text-xs font-semibold uppercase tracking-[0.18em] text-ink-2">
          Ready documents{readyDocuments.length > 0 ? ` · ${readyDocuments.length}` : ""}
        </p>

        {readyDocuments.length === 0 ? (
          <div className="rounded-xl border border-dashed border-white/10 py-8 text-center">
            <p className="text-sm text-ink-2">No processed documents yet.</p>
            <Link href={`/workspaces/${workspaceId}/documents`}
              className="mt-3 inline-block text-xs font-semibold text-accent-2 hover:text-accent">
              Upload documents →
            </Link>
          </div>
        ) : (
          <div className="grid gap-2 sm:grid-cols-2 xl:grid-cols-3">
            {readyDocuments.map((document) => {
              const checked = selectedDocumentIds.includes(document.id);
              return (
                <label
                  key={document.id}
                  className={`flex cursor-pointer items-center gap-3 rounded-xl border px-4 py-3 transition select-none ${
                    checked
                      ? "border-accent/40 bg-accent/10 text-accent-2"
                      : "border-white/8 bg-white/3 text-ink-2 hover:bg-white/6"
                  }`}
                >
                  <input
                    type="checkbox"
                    checked={checked}
                    onChange={() => toggleDocument(document.id)}
                    className="sr-only"
                  />
                  <span className={`flex h-4 w-4 shrink-0 items-center justify-center rounded border ${
                    checked ? "border-accent bg-accent text-white" : "border-white/20"
                  }`}>
                    {checked && (
                      <svg className="h-2.5 w-2.5" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth={3}>
                        <path strokeLinecap="round" strokeLinejoin="round" d="M5 13l4 4L19 7"/>
                      </svg>
                    )}
                  </span>
                  <span className="truncate text-sm font-medium">{document.filename}</span>
                </label>
              );
            })}
          </div>
        )}

        <div className="mt-5 flex flex-wrap items-center gap-3">
          <button
            type="button"
            onClick={triggerBuild}
            disabled={selectedDocumentIds.length < 2 || submitting}
            className="flex items-center gap-2 rounded-xl bg-accent px-5 py-2.5 text-sm font-semibold text-white transition hover:bg-accent-2 disabled:opacity-40"
          >
            {submitting ? (
              <>
                <svg className="h-4 w-4 animate-spin" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth={2}>
                  <path d="M12 2v4M12 18v4M4.93 4.93l2.83 2.83M16.24 16.24l2.83 2.83M2 12h4M18 12h4M4.93 19.07l2.83-2.83M16.24 7.76l2.83-2.83" strokeLinecap="round"/>
                </svg>
                Building…
              </>
            ) : (
              <>
                <svg className="h-4 w-4" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth={2}>
                  <path strokeLinecap="round" strokeLinejoin="round" d="M13 10V3L4 14h7v7l9-11h-7z"/>
                </svg>
                Build graph ({selectedDocumentIds.length} selected)
              </>
            )}
          </button>
          {selectedDocumentIds.length < 2 && (
            <p className="text-xs text-ink-3">Select at least 2 documents</p>
          )}
        </div>

        {error ? (
          <div className="mt-4 rounded-xl border border-danger/25 bg-danger/8 px-4 py-3 text-sm text-red-300">{error}</div>
        ) : null}
        {jobError ? (
          <div className="mt-4 rounded-xl border border-warning/25 bg-warning/8 px-4 py-3 text-sm text-yellow-300">Poll error: {jobError}</div>
        ) : null}
        {job ? (
          <div className="mt-4 flex items-center gap-3">
            <JobStatusBadge status={job.status} />
            <span className="text-xs font-mono text-ink-3">{job.id.slice(0, 20)}…</span>
          </div>
        ) : null}
      </div>

      {/* Graphs list */}
      <div className="glass rounded-2xl p-5">
        <p className="mb-4 text-xs font-semibold uppercase tracking-[0.18em] text-ink-2">Existing graphs</p>
        {loading ? (
          <div className="space-y-3">
            {[1, 2].map(i => <div key={i} className="h-16 animate-pulse rounded-xl bg-white/4" />)}
          </div>
        ) : graphs.length === 0 ? (
          <p className="text-sm text-ink-3">No graphs yet. Build one above.</p>
        ) : (
          <div className="space-y-2">
            {graphs.map((graph) => (
              <div key={graph.id} className="flex items-center justify-between gap-4 rounded-xl border border-white/6 bg-white/3 px-4 py-3">
                <div className="flex items-center gap-3">
                  <div className="flex h-8 w-8 items-center justify-center rounded-lg bg-accent/10">
                    <svg className="h-4 w-4 text-accent-2" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth={1.8}>
                      <circle cx="12" cy="5" r="2"/><circle cx="5" cy="19" r="2"/><circle cx="19" cy="19" r="2"/>
                      <path strokeLinecap="round" d="M12 7l-7 10M12 7l7 10"/>
                    </svg>
                  </div>
                  <div>
                    <p className="text-sm font-medium text-ink font-mono">
                      {graph.id.slice(0, 8)}<span className="text-ink-3">…</span>
                    </p>
                    <p className="text-[11px] text-ink-3 mt-0.5">
                      {new Date(graph.created_at).toLocaleDateString("en-US", { month: "short", day: "numeric", year: "numeric" })}
                    </p>
                  </div>
                </div>
                <div className="flex items-center gap-3">
                  <JobStatusBadge status={graph.status} />
                  {graph.status === "ready" && (
                    <Link
                      href={`/workspaces/${workspaceId}/graphs/${graph.id}`}
                      className="rounded-xl bg-accent/15 border border-accent/25 px-3 py-1.5 text-xs font-semibold text-accent-2 transition hover:bg-accent/25"
                    >
                      Open →
                    </Link>
                  )}
                </div>
              </div>
            ))}
          </div>
        )}
      </div>
    </div>
  );
}