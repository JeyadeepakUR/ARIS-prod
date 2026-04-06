type JobStatusBadgeProps = {
  status: string;
};

const statusStyle: Record<string, string> = {
  pending: "bg-sand text-ink",
  processing: "bg-amber-200 text-amber-900 pulse",
  ready: "bg-emerald-200 text-emerald-900",
  done: "bg-emerald-200 text-emerald-900",
  failed: "bg-red-200 text-red-900",
};

export function JobStatusBadge({ status }: JobStatusBadgeProps) {
  const style = statusStyle[status] ?? "bg-gray-200 text-gray-900";
  return (
    <span className={`inline-flex rounded-full px-3 py-1 text-xs font-semibold uppercase tracking-wide ${style}`}>
      {status}
    </span>
  );
}