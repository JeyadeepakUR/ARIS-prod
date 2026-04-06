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
        title="Upload And Track"
        description="Start an ingest job and watch pending, processing, and final status updates every 2 seconds."
      />

      <div className="mb-6 rounded-2xl border border-spice/20 bg-white/75 p-4">
        <div className="flex flex-col gap-3 md:flex-row md:items-center">
          <input
            type="file"
            accept=".txt,.pdf"
            onChange={(event) => {
              setSelectedFile(event.target.files?.[0] ?? null);
            }}
            className="w-full rounded-lg border border-spice/25 bg-white p-2 text-sm"
          />
          <button
            type="button"
            onClick={handleUpload}
            disabled={!selectedFile || submitting}
            className="rounded-lg bg-pine px-4 py-2 text-xs font-semibold uppercase tracking-wide text-white disabled:opacity-60"
          >
            {submitting ? "Starting..." : "Upload Document"}
          </button>
        </div>
        {message ? <p className="mt-3 text-sm text-ink/80">{message}</p> : null}
        {error ? <p className="mt-3 text-sm text-red-700">{error}</p> : null}
        {pollError ? <p className="mt-3 text-sm text-red-700">Polling error: {pollError}</p> : null}
      </div>

      <div className="mb-6 rounded-2xl border border-spice/20 bg-white/70 p-4">
        <p className="mb-2 text-xs font-semibold uppercase tracking-[0.2em] text-spice">Live Job Status</p>
        {job ? (
          <div className="flex flex-wrap items-center gap-2 text-sm">
            <JobStatusBadge status={job.status} />
            <span className="text-ink/70">Job ID: {job.id}</span>
          </div>
        ) : (
          <p className="text-sm text-ink/70">No active job. Upload a document to begin.</p>
        )}
      </div>

      <div className="rounded-2xl border border-spice/20 bg-white/65 p-4">
        <div className="mb-3 flex items-center justify-between">
          <p className="text-xs font-semibold uppercase tracking-[0.2em] text-spice">Recent Upload Jobs</p>
          <div className="flex items-center gap-3">
            <Link href={`/workspaces/${workspaceId}/graphs`} className="text-xs font-semibold uppercase tracking-wide text-spice">
              Go to Graphs
            </Link>
            <Link href="/workspaces" className="text-xs font-semibold uppercase tracking-wide text-pine">
              Back to Workspaces
            </Link>
          </div>
        </div>
        <div className="space-y-2">
          {records.map((record) => (
            <div key={record.jobId} className="rounded-lg border border-spice/15 bg-white/85 p-3 text-sm">
              <p className="font-semibold text-ink">{record.documentName}</p>
              <p className="text-xs text-ink/70">Job: {record.jobId}</p>
            </div>
          ))}
          {records.length === 0 ? <p className="text-sm text-ink/70">No uploads in this session.</p> : null}
        </div>
      </div>
    </div>
  );
}