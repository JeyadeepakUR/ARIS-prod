"use client";

import Link from "next/link";
import { useParams } from "next/navigation";
import { useState } from "react";

import { PageHeader } from "../../../../../components/layout/PageHeader";
import { JobStatusBadge } from "../../../../../components/shared/JobStatusBadge";
import { initializeUpload, uploadToSignedUrl } from "../../../../../lib/api/documents";
import { useJobPoller } from "../../../../../lib/hooks/useJobPoller";

type UploadRecord = {
  documentName: string;
  jobId: string;
  startedAt: string;
};

export default function WorkspaceDocumentsPage() {
  const params = useParams<{ id: string }>();
  const workspaceId = params.id;

  const [selectedFile, setSelectedFile] = useState<File | null>(null);
  const [activeJobId, setActiveJobId] = useState<string | null>(null);
  const [records, setRecords] = useState<UploadRecord[]>([]);
  const [message, setMessage] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [submitting, setSubmitting] = useState(false);

  const { job, error: pollError } = useJobPoller(activeJobId, Boolean(activeJobId));

  async function handleUpload() {
    if (!selectedFile || !workspaceId) {
      return;
    }

    setSubmitting(true);
    setError(null);
    setMessage(null);

    try {
      const initialized = await initializeUpload(workspaceId, selectedFile);
      setActiveJobId(initialized.job_id);
      setRecords((prev) => [
        {
          documentName: selectedFile.name,
          jobId: initialized.job_id,
          startedAt: new Date().toISOString(),
        },
        ...prev,
      ]);

      try {
        await uploadToSignedUrl(initialized.upload_url, selectedFile);
        setMessage("File uploaded. Ingestion is running.");
      } catch {
        // The backend supports fallback ingest for local environments without S3.
        setMessage("Upload URL failed in local mode. Ingestion job still queued using fallback content.");
      }
    } catch (uploadError) {
      setError(uploadError instanceof Error ? uploadError.message : "Upload could not be initialized");
    } finally {
      setSubmitting(false);
    }
  }

  return (
    <div>
      <PageHeader
        eyebrow="Documents"
        title="Upload & Ingest"
        description="Upload .txt or .pdf files. ARIS ingests the content and prepares it for graph building."
        action={
          <Link
            href={`/workspaces/${workspaceId}/graphs`}
            className="flex items-center gap-2 rounded-xl bg-accent/12 border border-accent/20 px-4 py-2 text-xs font-semibold text-accent-2 transition hover:bg-accent/20"
          >
            Build Graph →
          </Link>
        }
      />

      {/* Upload zone */}
      <div className="mb-6 rounded-2xl border border-dashed border-white/12 bg-white/3 p-6 transition hover:border-accent/30 hover:bg-white/5">
        <div className="flex flex-col items-center gap-4 text-center">
          <svg className="h-10 w-10 text-ink-3" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={1.2}>
            <path strokeLinecap="round" strokeLinejoin="round" d="M4 16v2a2 2 0 002 2h12a2 2 0 002-2v-2M16 12l-4-4m0 0L8 12m4-4v12"/>
          </svg>
          <div>
            <p className="text-sm font-medium text-ink">Drop a file or click to browse</p>
            <p className="mt-1 text-xs text-ink-3">.txt or .pdf · max 10 MB</p>
          </div>
          <label className="cursor-pointer">
            <input
              type="file"
              accept=".txt,.pdf"
              className="sr-only"
              onChange={(event) => setSelectedFile(event.target.files?.[0] ?? null)}
            />
            <span className="rounded-xl border border-white/12 bg-white/6 px-5 py-2 text-xs font-semibold text-ink-2 transition hover:bg-white/10 hover:text-ink">
              {selectedFile ? selectedFile.name : "Choose file"}
            </span>
          </label>
          {selectedFile && (
            <button
              type="button"
              onClick={handleUpload}
              disabled={submitting}
              className="rounded-xl bg-accent px-6 py-2.5 text-sm font-semibold text-white transition hover:bg-accent-2 disabled:opacity-50"
            >
              {submitting ? "Uploading…" : `Upload "${selectedFile.name}"`}
            </button>
          )}
        </div>

        {message ? (
          <div className="mt-4 rounded-xl border border-success/25 bg-success/8 px-4 py-3 text-sm text-emerald-300">{message}</div>
        ) : null}
        {error ? (
          <div className="mt-4 rounded-xl border border-danger/25 bg-danger/8 px-4 py-3 text-sm text-red-300">{error}</div>
        ) : null}
        {pollError ? (
          <div className="mt-4 rounded-xl border border-warning/25 bg-warning/8 px-4 py-3 text-sm text-yellow-300">Polling: {pollError}</div>
        ) : null}
      </div>

      {/* Active job */}
      {job ? (
        <div className="mb-6 glass rounded-2xl p-4">
          <div className="flex items-center justify-between">
            <p className="text-xs font-semibold uppercase tracking-[0.18em] text-ink-2">Active job</p>
            <JobStatusBadge status={job.status} />
          </div>
          <p className="mt-2 text-xs font-mono text-ink-3">{job.id}</p>
        </div>
      ) : null}

      {/* Upload history */}
      <div className="glass rounded-2xl p-4">
        <p className="mb-3 text-xs font-semibold uppercase tracking-[0.18em] text-ink-2">Session uploads</p>
        {records.length === 0 ? (
          <p className="text-sm text-ink-3">No uploads yet this session.</p>
        ) : (
          <div className="space-y-2">
            {records.map((record) => (
              <div key={record.jobId} className="flex items-center justify-between rounded-xl border border-white/6 bg-white/3 px-4 py-3">
                <div>
                  <p className="text-sm font-medium text-ink">{record.documentName}</p>
                  <p className="text-xs text-ink-3 mt-0.5 font-mono">{record.jobId.slice(0, 16)}…</p>
                </div>
                <svg className="h-4 w-4 text-success" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}>
                  <path strokeLinecap="round" strokeLinejoin="round" d="M5 13l4 4L19 7"/>
                </svg>
              </div>
            ))}
          </div>
        )}
      </div>
    </div>
  );
}