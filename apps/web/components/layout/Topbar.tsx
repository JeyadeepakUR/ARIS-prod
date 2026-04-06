"use client";

import { useRouter } from "next/navigation";

import { useAuth } from "../../lib/hooks/useAuth";

export function Topbar() {
  const { user, logout } = useAuth();
  const router = useRouter();

  return (
    <header className="glass-panel flex items-center justify-between rounded-2xl px-4 py-3">
      <div>
        <p className="text-xs uppercase tracking-[0.2em] text-spice">ARIS Console</p>
        <p className="text-sm font-semibold text-ink">{user?.email ?? "Anonymous"}</p>
      </div>
      <button
        onClick={() => {
          logout();
          router.replace("/login");
        }}
        className="rounded-lg border border-spice/25 bg-white/80 px-3 py-2 text-xs font-semibold uppercase tracking-wide text-spice hover:bg-sand"
      >
        Logout
      </button>
    </header>
  );
}