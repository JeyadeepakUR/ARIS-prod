"use client";

import { useRouter } from "next/navigation";

import { useAuth } from "../../lib/hooks/useAuth";

export function Topbar() {
  const { user, logout } = useAuth();
  const router = useRouter();

  return (
    <header className="glass flex items-center justify-between rounded-2xl px-4 py-3">
      <div className="flex items-center gap-3">
        <span className="flex h-8 w-8 items-center justify-center rounded-lg bg-accent/10 text-accent-2 text-sm font-bold">
          {user?.email?.[0]?.toUpperCase() ?? "A"}
        </span>
        <div>
          <p className="text-[11px] font-medium text-ink-2">Signed in as</p>
          <p className="text-sm font-semibold text-ink leading-none">{user?.email ?? "Anonymous"}</p>
        </div>
      </div>

      <div className="flex items-center gap-3">
        <div className="flex items-center gap-1.5">
          <span className="h-2 w-2 rounded-full bg-success animate-pulse" />
          <span className="text-[11px] text-ink-2 font-medium">API Connected</span>
        </div>
        <button
          onClick={() => {
            logout();
            router.replace("/login");
          }}
          className="rounded-xl border border-white/10 bg-white/5 px-3 py-1.5 text-xs font-semibold text-ink-2 transition hover:bg-white/10 hover:text-ink"
        >
          Sign out
        </button>
      </div>
    </header>
  );
}