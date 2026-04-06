"use client";

import Link from "next/link";
import { usePathname } from "next/navigation";

const links = [
  { href: "/workspaces", label: "Workspaces" },
  { href: "/", label: "Landing" },
];

export function Sidebar() {
  const pathname = usePathname();

  return (
    <aside className="glass-panel sticky top-4 h-fit rounded-2xl p-4">
      <p className="text-xs font-semibold uppercase tracking-[0.2em] text-spice">Workspace</p>
      <nav className="mt-4 flex flex-col gap-2">
        {links.map((link) => {
          const active = pathname === link.href || pathname.startsWith(`${link.href}/`);
          return (
            <Link
              key={link.href}
              href={link.href}
              className={`rounded-lg px-3 py-2 text-sm font-medium transition ${
                active ? "bg-pine text-white" : "bg-white/70 text-ink hover:bg-sand"
              }`}
            >
              {link.label}
            </Link>
          );
        })}
      </nav>
    </aside>
  );
}