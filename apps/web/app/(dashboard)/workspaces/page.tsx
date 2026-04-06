"use client";

import Link from "next/link";
import { FormEvent, useEffect, useState } from "react";

import { PageHeader } from "../../../components/layout/PageHeader";
import { createWorkspace, listWorkspaces, type Workspace } from "../../../lib/api/workspaces";

function slugify(value: string): string {
  return value
    .trim()
    .toLowerCase()
    .replace(/[^a-z0-9]+/g, "-")
    .replace(/^-+|-+$/g, "")
    .slice(0, 40);
}

export default function WorkspacesPage() {
  const [workspaces, setWorkspaces] = useState<Workspace[]>([]);
  const [name, setName] = useState("");
  const [error, setError] = useState<string | null>(null);
  const [loading, setLoading] = useState(true);
  const [submitting, setSubmitting] = useState(false);

  useEffect(() => {
    let mounted = true;
    listWorkspaces()
      .then((items) => {
        if (mounted) {
          setWorkspaces(items);
        }
      })
      .catch((loadError) => {
        if (mounted) {
          setError(loadError instanceof Error ? loadError.message : "Unable to load workspaces");
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
  }, []);

  async function onCreateWorkspace(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();
    const trimmed = name.trim();
    if (!trimmed) {
      return;
    }

    setSubmitting(true);
    setError(null);
    try {
      const candidateSlug = `${slugify(trimmed)}-${Math.floor(Date.now() / 1000)}`;
      const created = await createWorkspace(trimmed, candidateSlug);
      setWorkspaces((prev) => [created, ...prev]);
      setName("");
    } catch (createError) {
      setError(createError instanceof Error ? createError.message : "Unable to create workspace");
    } finally {
      setSubmitting(false);
    }
  }

  return (
    <div>
      <PageHeader
        eyebrow="Workspace Hub"
        title="Your Workspaces"
        description="Create a workspace, then upload documents and monitor ingestion in real time."
      />

      <form onSubmit={onCreateWorkspace} className="mb-6 grid gap-3 md:grid-cols-[1fr_auto]">
        <input
          value={name}
          onChange={(event) => setName(event.target.value)}
          placeholder="Workspace name"
          className="rounded-xl border border-spice/20 bg-white/90 px-4 py-3 text-sm"
        />
        <button
          type="submit"
          disabled={submitting}
          className="rounded-xl bg-spice px-4 py-3 text-sm font-semibold uppercase tracking-wide text-white disabled:opacity-60"
        >
          {submitting ? "Creating..." : "Create Workspace"}
        </button>
      </form>

      {error ? <p className="mb-3 text-sm text-red-700">{error}</p> : null}
      {loading ? <p className="text-sm text-ink/70">Loading workspaces...</p> : null}

      <div className="grid gap-3">
        {workspaces.map((workspace) => (
          <div key={workspace.id} className="rounded-xl border border-spice/20 bg-white/75 p-4">
            <div className="flex flex-wrap items-center justify-between gap-2">
              <div>
                <p className="text-sm font-semibold text-ink">{workspace.name}</p>
                <p className="text-xs uppercase tracking-wide text-ink/65">{workspace.slug}</p>
              </div>
              <div className="flex gap-2">
                <Link
                  href={`/workspaces/${workspace.id}/documents`}
                  className="rounded-lg border border-pine/20 bg-pine px-3 py-2 text-xs font-semibold uppercase tracking-wide text-white"
                >
                  Documents
                </Link>
                <Link
                  href={`/workspaces/${workspace.id}/graphs`}
                  className="rounded-lg border border-spice/20 bg-spice px-3 py-2 text-xs font-semibold uppercase tracking-wide text-white"
                >
                  Graphs
                </Link>
              </div>
            </div>
          </div>
        ))}

        {!loading && workspaces.length === 0 ? (
          <p className="rounded-xl border border-dashed border-spice/30 bg-white/60 p-6 text-sm text-ink/75">
            No workspaces yet. Create one to begin.
          </p>
        ) : null}
      </div>
    </div>
  );
}