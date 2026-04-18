import Link from "next/link";

export default function HomePage() {
  return (
    <main className="relative flex min-h-screen flex-col items-center justify-center overflow-hidden px-6 py-20">
      {/* Background glow blobs */}
      <div className="pointer-events-none absolute inset-0 overflow-hidden">
        <div className="absolute -left-40 -top-40 h-[600px] w-[600px] rounded-full bg-accent/10 blur-[120px]" />
        <div className="absolute -right-40 bottom-0 h-[500px] w-[500px] rounded-full bg-bridge/8 blur-[100px]" />
      </div>

      <div className="relative z-10 w-full max-w-4xl">
        {/* Brand */}
        <div className="mb-8 flex items-center gap-3">
          <span className="flex h-10 w-10 items-center justify-center rounded-xl bg-accent/15 text-accent">
            <svg className="h-5 w-5" viewBox="0 0 24 24" fill="currentColor">
              <path d="M12 2a10 10 0 1 0 10 10A10 10 0 0 0 12 2zm1 17.93V18a1 1 0 0 0-2 0v1.93A8 8 0 0 1 4.07 13H6a1 1 0 0 0 0-2H4.07A8 8 0 0 1 11 4.07V6a1 1 0 0 0 2 0V4.07A8 8 0 0 1 19.93 11H18a1 1 0 0 0 0 2h1.93A8 8 0 0 1 13 19.93z"/>
            </svg>
          </span>
          <span className="text-lg font-bold text-ink tracking-tight">ARIS</span>
          <span className="ml-2 rounded-full border border-accent/30 bg-accent/10 px-2.5 py-0.5 text-[10px] font-semibold uppercase tracking-wider text-accent-2">
            Alpha
          </span>
        </div>

        {/* Hero */}
        <h1 className="max-w-3xl text-5xl font-bold leading-tight text-ink md:text-6xl">
          Find what your research{" "}
          <span className="bg-gradient-to-r from-accent-2 to-bridge bg-clip-text text-transparent">
            implies
          </span>{" "}
          before anyone else does.
        </h1>
        <p className="mt-6 max-w-xl text-lg text-ink-2 leading-relaxed">
          Upload papers. ARIS constructs a knowledge graph, surfaces bridge concepts across domains, and generates falsifiable hypotheses — all in seconds.
        </p>

        <div className="mt-10 flex flex-wrap gap-4">
          <Link
            href="/register"
            className="rounded-xl bg-accent px-7 py-3.5 text-sm font-semibold text-white shadow-glow transition hover:bg-accent-2"
          >
            Get started free
          </Link>
          <Link
            href="/login"
            className="rounded-xl border border-white/12 bg-white/5 px-7 py-3.5 text-sm font-semibold text-ink-2 transition hover:bg-white/10 hover:text-ink"
          >
            Sign in
          </Link>
        </div>

        {/* Feature strip */}
        <div className="mt-16 grid grid-cols-2 gap-4 sm:grid-cols-4">
          {[
            { icon: "⬡", title: "Knowledge Graph", desc: "Domain concept network built from your documents" },
            { icon: "⬡", title: "Bridge Concepts", desc: "Cross-domain connections with evidence" },
            { icon: "⬡", title: "Hypotheses", desc: "LLM-generated, falsifiable research claims" },
            { icon: "⬡", title: "Research Plans", desc: "Prioritised actions for follow-up work" },
          ].map((f) => (
            <div key={f.title} className="glass rounded-2xl p-4">
              <span className="text-2xl text-accent">{f.icon}</span>
              <p className="mt-2 text-sm font-semibold text-ink">{f.title}</p>
              <p className="mt-1 text-xs text-ink-2 leading-relaxed">{f.desc}</p>
            </div>
          ))}
        </div>
      </div>
    </main>
  );
}