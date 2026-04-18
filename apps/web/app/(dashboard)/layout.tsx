"use client";

import { useRouter } from "next/navigation";
import { useEffect } from "react";

import { Sidebar } from "../../components/layout/Sidebar";
import { Topbar } from "../../components/layout/Topbar";
import { useAuth } from "../../lib/hooks/useAuth";

export default function DashboardLayout({ children }: { children: React.ReactNode }) {
  const { isReady, user } = useAuth();
  const router = useRouter();

  useEffect(() => {
    if (isReady && !user) {
      router.replace("/login");
    }
  }, [isReady, router, user]);

  if (!isReady || !user) {
    return (
      <main className="flex min-h-screen items-center justify-center">
        <p className="glass-panel rounded-xl px-4 py-3 text-sm font-semibold text-spice">Loading dashboard...</p>
      </main>
    );
  }

  return (
    <div className="mx-auto grid min-h-screen w-full max-w-[1600px] grid-cols-1 gap-4 px-3 py-4 lg:px-5 md:grid-cols-[220px_1fr]">
      <Sidebar />
      <section className="flex min-h-0 flex-col gap-4">
        <Topbar />
        <div className="glass rounded-2xl p-5 md:p-7 flex-1">{children}</div>
      </section>
    </div>
  );
}