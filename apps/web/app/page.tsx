import Link from "next/link";

export default function HomePage() {
  return (
    <main className="mx-auto flex min-h-screen w-full max-w-5xl flex-col justify-center px-6 py-16">
      <section className="glass-panel rounded-3xl p-10 md:p-14">
        <p className="text-sm font-semibold uppercase tracking-[0.2em] text-spice">ARIS Platform</p>
        <h1 className="mt-3 max-w-2xl text-4xl font-bold leading-tight text-ink md:text-6xl">
          Find what your documents imply before your competitors do.
        </h1>
        <p className="mt-6 max-w-2xl text-base leading-relaxed text-ink/80 md:text-lg">
          Upload research assets, let ARIS connect the graph, and investigate every edge with evidence.
        </p>
        <div className="mt-9 flex flex-wrap gap-4">
          <Link
            href="/register"
            className="rounded-xl bg-pine px-6 py-3 text-sm font-semibold uppercase tracking-wide text-white transition hover:translate-y-[-1px] hover:bg-[#23483d]"
          >
            Create Account
          </Link>
          <Link
            href="/login"
            className="rounded-xl border border-spice/30 bg-white/80 px-6 py-3 text-sm font-semibold uppercase tracking-wide text-spice transition hover:translate-y-[-1px]"
          >
            Sign In
          </Link>
        </div>
      </section>
    </main>
  );
}