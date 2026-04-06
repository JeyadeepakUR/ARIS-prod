"use client";

import Link from "next/link";
import { useRouter } from "next/navigation";
import { FormEvent, useState } from "react";

import { useAuth } from "../../../lib/hooks/useAuth";

export default function LoginPage() {
  const router = useRouter();
  const { login } = useAuth();
  const [email, setEmail] = useState("");
  const [password, setPassword] = useState("");
  const [error, setError] = useState<string | null>(null);
  const [submitting, setSubmitting] = useState(false);

  async function onSubmit(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();
    setSubmitting(true);
    setError(null);
    try {
      await login(email, password);
      router.replace("/workspaces");
    } catch (submitError) {
      setError(submitError instanceof Error ? submitError.message : "Unable to sign in");
    } finally {
      setSubmitting(false);
    }
  }

  return (
    <main className="mx-auto flex min-h-screen w-full max-w-md items-center px-6 py-16">
      <section className="glass-panel w-full rounded-3xl p-8">
        <p className="text-xs font-semibold uppercase tracking-[0.2em] text-spice">Welcome back</p>
        <h1 className="mt-2 text-3xl font-bold text-ink">Sign in to ARIS</h1>
        <form className="mt-7 space-y-4" onSubmit={onSubmit}>
          <input
            required
            type="email"
            value={email}
            onChange={(event) => setEmail(event.target.value)}
            placeholder="you@example.com"
            className="w-full rounded-xl border border-spice/20 bg-white/90 px-4 py-3 text-sm"
          />
          <input
            required
            minLength={8}
            type="password"
            value={password}
            onChange={(event) => setPassword(event.target.value)}
            placeholder="Password"
            className="w-full rounded-xl border border-spice/20 bg-white/90 px-4 py-3 text-sm"
          />
          {error ? <p className="text-sm text-red-700">{error}</p> : null}
          <button
            type="submit"
            disabled={submitting}
            className="w-full rounded-xl bg-pine px-4 py-3 text-sm font-semibold uppercase tracking-wide text-white disabled:opacity-60"
          >
            {submitting ? "Signing in..." : "Sign in"}
          </button>
        </form>
        <p className="mt-5 text-sm text-ink/75">
          Need an account? <Link href="/register" className="font-semibold text-spice">Register</Link>
        </p>
      </section>
    </main>
  );
}