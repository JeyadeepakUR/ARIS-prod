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
        description="Create a workspace to upload documents and build knowledge graphs."
      />

      {/* Create form */}
      <form onSubmit={onCreateWorkspace} className="mb-8 flex items-center gap-3">
        <input
          value={name}
          onChange={(event) => setName(event.target.value)}
          placeholder="New workspace name…"
          className="flex-1 rounded-xl border border-white/10 bg-white/5 px-4 py-2.5 text-sm text-ink placeholder:text-ink-3 transition focus:border-accent focus:bg-white/8"
        />
        <button
          type="submit"
          disabled={submitting || !name.trim()}
          className="flex items-center gap-2 rounded-xl bg-accent px-4 py-2.5 text-sm font-semibold text-white transition hover:bg-accent-2 disabled:opacity-40"
        >
          <svg className="h-4 w-4" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}>
            <path strokeLinecap="round" strokeLinejoin="round" d="M12 4v16m8-8H4" />
          </svg>
          {submitting ? "Creating…" : "Create"}
        </button>
      </form>

      {error ? (
        <div className="mb-4 rounded-xl border border-danger/30 bg-danger/10 px-4 py-3 text-sm text-red-300">{error}</div>
      ) : null}

      {/* Workspaces grid */}
      {loading ? (
        <div className="grid gap-4 sm:grid-cols-2 xl:grid-cols-3">
          {[1, 2].map((i) => (
            <div key={i} className="h-40 animate-pulse rounded-2xl bg-white/4 border border-white/6" />
          ))}
        </div>
      ) : (
        <div className="grid gap-4 sm:grid-cols-2 xl:grid-cols-3">
          {workspaces.map((workspace) => (
            <div key={workspace.id} className="glass rounded-2xl p-5 transition hover:border-white/14 group">
              <div className="flex items-start justify-between gap-2">
                <div className="flex items-center gap-3">
                  <span className="flex h-9 w-9 items-center justify-center rounded-xl bg-accent/10 text-accent-2 text-sm font-bold">
                    {workspace.name[0]?.toUpperCase()}
                  </span>
                  <div>
                    <p className="font-semibold text-ink text-sm">{workspace.name}</p>
                    <p className="text-[11px] text-ink-3 mt-0.5">{workspace.slug}</p>
                  </div>
                </div>
                <span className="flex h-5 w-5 items-center justify-center rounded-full border border-success/30 bg-success/10">
                  <span className="h-1.5 w-1.5 rounded-full bg-success" />
                </span>
              </div>

              <div className="mt-5 flex gap-2">
                <Link
                  href={`/workspaces/${workspace.id}/documents`}
                  className="flex-1 rounded-xl bg-white/6 border border-white/8 py-2 text-center text-xs font-semibold text-ink-2 transition hover:bg-white/10 hover:text-ink"
                >
                  Documents
                </Link>
                <Link
                  href={`/workspaces/${workspace.id}/graphs`}
                  className="flex-1 rounded-xl bg-accent/12 border border-accent/20 py-2 text-center text-xs font-semibold text-accent-2 transition hover:bg-accent/20"
                >
                  Graphs
                </Link>
              </div>
            </div>
          ))}

          {workspaces.length === 0 ? (
            <div className="col-span-full rounded-2xl border border-dashed border-white/10 bg-white/2 p-10 text-center">
              <p className="text-sm text-ink-2">No workspaces yet. Create one above to begin.</p>
            </div>
          ) : null}
        </div>
      )}
    </div>
  );
}