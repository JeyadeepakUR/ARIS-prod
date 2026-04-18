"use client";

import Link from "next/link";
import { usePathname } from "next/navigation";

const NAV = [
  {
    href: "/workspaces",
    label: "Workspaces",
    icon: (
      <svg className="h-4 w-4 shrink-0" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={1.8}>
        <path strokeLinecap="round" strokeLinejoin="round" d="M3 7h18M3 12h18M3 17h18" />
      </svg>
    ),
  },
];

export function Sidebar() {
  const pathname = usePathname();

  return (
    <aside className="sticky top-4 flex h-fit flex-col gap-1 rounded-2xl glass p-3">
      {/* Brand */}
      <div className="mb-3 flex items-center gap-2 px-2 pt-1">
        <span className="flex h-7 w-7 items-center justify-center rounded-lg bg-accent/20 text-accent">
          <svg className="h-4 w-4" viewBox="0 0 24 24" fill="currentColor">
            <path d="M12 2a10 10 0 1 0 10 10A10 10 0 0 0 12 2zm1 17.93V18a1 1 0 0 0-2 0v1.93A8 8 0 0 1 4.07 13H6a1 1 0 0 0 0-2H4.07A8 8 0 0 1 11 4.07V6a1 1 0 0 0 2 0V4.07A8 8 0 0 1 19.93 11H18a1 1 0 0 0 0 2h1.93A8 8 0 0 1 13 19.93z"/>
          </svg>
        </span>
        <span className="text-sm font-bold tracking-tight text-ink">ARIS</span>
      </div>

      <nav className="flex flex-col gap-0.5">
        {NAV.map((item) => {
          const active = pathname.startsWith(item.href);
          return (
            <Link
              key={item.href}
              href={item.href}
              className={`group flex items-center gap-2.5 rounded-xl px-3 py-2.5 text-sm font-medium transition-all duration-150 ${
                active
                  ? "bg-accent/15 text-accent-2"
                  : "text-ink-2 hover:bg-white/5 hover:text-ink"
              }`}
            >
              {item.icon}
              {item.label}
              {active && (
                <span className="ml-auto h-1.5 w-1.5 rounded-full bg-accent" />
              )}
            </Link>
          );
        })}
      </nav>

      <div className="mt-3 border-t border-white/[0.06] pt-3">
        <div className="rounded-xl bg-accent/8 border border-accent/15 px-3 py-2.5">
          <p className="text-[10px] font-semibold uppercase tracking-wider text-accent-2">Intelligence</p>
          <p className="mt-0.5 text-[11px] text-ink-2">Knowledge graph active</p>
        </div>
      </div>
    </aside>
  );
}