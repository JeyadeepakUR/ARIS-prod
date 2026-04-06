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
        title="Build Domain Concept Network"
        description="Separate concepts by domain and connect domains only through explicit bridge concepts discovered in the same paper."
      />

      <div className="mb-6 rounded-2xl border border-spice/20 bg-white/75 p-4">
        <p className="mb-3 text-xs font-semibold uppercase tracking-[0.2em] text-spice">Ready Documents</p>
        <div className="grid gap-2">
          {readyDocuments.map((document) => (
            <label key={document.id} className="flex items-center gap-3 rounded-lg border border-spice/15 bg-white p-3 text-sm">
              <input
                type="checkbox"
                checked={selectedDocumentIds.includes(document.id)}
                onChange={() => toggleDocument(document.id)}
              />
              <span className="font-medium text-ink">{document.filename}</span>
            </label>
          ))}
          {readyDocuments.length === 0 ? (
            <p className="text-sm text-ink/70">No ready documents found. Upload and process documents first.</p>
          ) : null}
        </div>

        <div className="mt-4 flex flex-wrap items-center gap-3">
          <button
            type="button"
            onClick={triggerBuild}
            disabled={selectedDocumentIds.length < 2 || submitting}
            className="rounded-lg bg-spice px-4 py-2 text-xs font-semibold uppercase tracking-wide text-white disabled:opacity-50"
          >
            {submitting ? "Starting build..." : "Build Graph"}
          </button>
          <Link href={`/workspaces/${workspaceId}/documents`} className="text-xs font-semibold uppercase tracking-wide text-pine">
            Go to Documents
          </Link>
        </div>

        {error ? <p className="mt-3 text-sm text-red-700">{error}</p> : null}
        {jobError ? <p className="mt-3 text-sm text-red-700">Job polling error: {jobError}</p> : null}
        {job ? (
          <div className="mt-3 flex items-center gap-2">
            <JobStatusBadge status={job.status} />
            <span className="text-xs text-ink/70">Job {job.id}</span>
          </div>
        ) : null}
      </div>

      <div className="rounded-2xl border border-spice/20 bg-white/75 p-4">
        <p className="mb-3 text-xs font-semibold uppercase tracking-[0.2em] text-spice">Graphs</p>
        {loading ? <p className="text-sm text-ink/70">Loading graphs...</p> : null}
        <div className="space-y-2">
          {graphs.map((graph) => (
            <div key={graph.id} className="flex flex-wrap items-center justify-between gap-2 rounded-lg border border-spice/15 bg-white p-3">
              <div>
                <p className="text-sm font-semibold text-ink">Graph {graph.id.slice(0, 8)}</p>
                <p className="text-xs uppercase tracking-wide text-ink/65">{graph.status}</p>
              </div>
              <Link
                href={`/workspaces/${workspaceId}/graphs/${graph.id}`}
                className="rounded-lg bg-pine px-3 py-2 text-xs font-semibold uppercase tracking-wide text-white"
              >
                Open Canvas
              </Link>
            </div>
          ))}
          {!loading && graphs.length === 0 ? (
            <p className="text-sm text-ink/70">No graphs yet.</p>
          ) : null}
        </div>
      </div>
    </div>
  );
}