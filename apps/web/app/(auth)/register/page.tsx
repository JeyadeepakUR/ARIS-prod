"use client";

import Link from "next/link";
import { useRouter } from "next/navigation";
import { FormEvent, useState } from "react";

import { useAuth } from "../../../lib/hooks/useAuth";

export default function RegisterPage() {
  const router = useRouter();
  const { register } = useAuth();
  const [email, setEmail] = useState("");
  const [password, setPassword] = useState("");
  const [error, setError] = useState<string | null>(null);
  const [submitting, setSubmitting] = useState(false);

  async function onSubmit(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();
    setSubmitting(true);
    setError(null);

    try {
      await register(email, password);
      router.replace("/workspaces");
    } catch (submitError) {
      setError(submitError instanceof Error ? submitError.message : "Unable to register");
    } finally {
      setSubmitting(false);
    }
  }

  return (
    <div className="flex min-h-screen">
      {/* Left panel */}
      <div className="hidden w-1/2 flex-col justify-between bg-bg-2 p-12 lg:flex" style={{background:"linear-gradient(135deg,#0d0f1a 0%,#111827 100%)"}}>
        <div className="flex items-center gap-2.5">
          <span className="flex h-8 w-8 items-center justify-center rounded-lg bg-accent/20 text-accent">
            <svg className="h-4 w-4" viewBox="0 0 24 24" fill="currentColor">
              <path d="M12 2a10 10 0 1 0 10 10A10 10 0 0 0 12 2zm1 17.93V18a1 1 0 0 0-2 0v1.93A8 8 0 0 1 4.07 13H6a1 1 0 0 0 0-2H4.07A8 8 0 0 1 11 4.07V6a1 1 0 0 0 2 0V4.07A8 8 0 0 1 19.93 11H18a1 1 0 0 0 0 2h1.93A8 8 0 0 1 13 19.93z"/>
            </svg>
          </span>
          <span className="text-base font-bold text-ink">ARIS</span>
        </div>
        <div>
          <h2 className="text-4xl font-bold leading-tight text-ink">Start discovering<br />hidden connections.</h2>
          <p className="mt-4 text-base text-ink-2 leading-relaxed max-w-sm">Join ARIS and let the knowledge graph surface relationships your team would take weeks to find manually.</p>
          <ul className="mt-8 space-y-3">
            {[
              "Upload PDF or text documents",
              "Auto-build domain concept networks",
              "Discover cross-domain bridge concepts",
              "Generate falsifiable research hypotheses",
            ].map(f=>(
              <li key={f} className="flex items-center gap-2.5 text-sm text-ink-2">
                <span className="flex h-5 w-5 shrink-0 items-center justify-center rounded-full bg-success/20 text-success text-xs">✓</span>
                {f}
              </li>
            ))}
          </ul>
        </div>
        <p className="text-xs text-ink-3">Autonomous Research Intelligence System · 2025</p>
      </div>

      {/* Right panel */}
      <main className="flex w-full items-center justify-center px-6 py-16 lg:w-1/2">
        <div className="w-full max-w-sm">
          <p className="text-[11px] font-semibold uppercase tracking-[0.22em] text-accent-2">Get started</p>
          <h1 className="mt-2 text-2xl font-bold text-ink">Create your account</h1>

          <form className="mt-8 space-y-4" onSubmit={onSubmit}>
            <div>
              <label className="mb-1.5 block text-xs font-medium text-ink-2">Email</label>
              <input
                required
                type="email"
                value={email}
                onChange={(event) => setEmail(event.target.value)}
                placeholder="you@example.com"
                className="w-full rounded-xl border border-white/10 bg-white/5 px-4 py-3 text-sm text-ink placeholder:text-ink-3 transition focus:border-accent focus:bg-white/8"
              />
            </div>
            <div>
              <label className="mb-1.5 block text-xs font-medium text-ink-2">Password</label>
              <input
                required
                minLength={8}
                type="password"
                value={password}
                onChange={(event) => setPassword(event.target.value)}
                placeholder="At least 8 characters"
                className="w-full rounded-xl border border-white/10 bg-white/5 px-4 py-3 text-sm text-ink placeholder:text-ink-3 transition focus:border-accent focus:bg-white/8"
              />
            </div>

            {error ? (
              <div className="rounded-xl border border-danger/30 bg-danger/10 px-4 py-3 text-sm text-red-300">
                {error}
              </div>
            ) : null}

            <button
              type="submit"
              disabled={submitting}
              className="w-full rounded-xl bg-accent px-4 py-3 text-sm font-semibold text-white transition hover:bg-accent-2 disabled:opacity-50"
            >
              {submitting ? "Creating account…" : "Create account"}
            </button>
          </form>

          <p className="mt-6 text-sm text-ink-2">
            Already have an account?{" "}
            <Link href="/login" className="font-semibold text-accent-2 hover:text-accent">
              Sign in
            </Link>
          </p>
        </div>
      </main>
    </div>
  );
}