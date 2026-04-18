"use client";

type JobStatusBadgeProps = {
  status: string;
};

const statusStyle: Record<string, { cls: string; dot: string }> = {
  pending: { cls: "border-white/10 bg-white/6 text-ink-2", dot: "bg-ink-3" },
  processing: { cls: "border-warning/30 bg-warning/10 text-yellow-300", dot: "bg-warning pulse" },
  ready: { cls: "border-success/30 bg-success/10 text-emerald-300", dot: "bg-success" },
  done: { cls: "border-success/30 bg-success/10 text-emerald-300", dot: "bg-success" },
  failed: { cls: "border-danger/30 bg-danger/10 text-red-300", dot: "bg-danger" },
};

export function JobStatusBadge({ status }: JobStatusBadgeProps) {
  const s = statusStyle[status] ?? { cls: "border-white/10 bg-white/6 text-ink-2", dot: "bg-ink-3" };
  return (
    <span className={`inline-flex items-center gap-1.5 rounded-full border px-2.5 py-1 text-xs font-semibold uppercase tracking-wide ${s.cls}`}>
      <span className={`h-1.5 w-1.5 rounded-full ${s.dot}`} />
      {status}
    </span>
  );
}