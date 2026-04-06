import { Handle, Position } from "reactflow";

type NodeCardProps = {
  label: string;
  nodeType: string;
};

export function NodeCard({ label, nodeType }: NodeCardProps) {
  const styleByType: Record<string, { card: string; badge: string }> = {
    document: {
      card: "border-spice/35 bg-white",
      badge: "bg-spice/15 text-spice",
    },
    domain: {
      card: "border-pine/45 bg-emerald-50/50",
      badge: "bg-pine/20 text-pine",
    },
    concept: {
      card: "border-slate-300/70 bg-white",
      badge: "bg-slate-200/80 text-slate-700",
    },
    bridge_concept: {
      card: "border-amber-400/70 bg-amber-50/60",
      badge: "bg-amber-200/80 text-amber-900",
    },
  };

  const cardStyle = styleByType[nodeType]?.card ?? "border-spice/30 bg-white/95";
  const badgeStyle = styleByType[nodeType]?.badge ?? "bg-sand text-ink";

  return (
    <div className={`min-w-[190px] rounded-xl border p-3 shadow-sm ${cardStyle}`}>
      <Handle type="target" position={Position.Left} className="!h-2.5 !w-2.5 !bg-spice" />
      <Handle type="source" position={Position.Right} className="!h-2.5 !w-2.5 !bg-pine" />
      <p className="line-clamp-2 text-[13px] font-semibold leading-snug text-ink">{label}</p>
      <span className={`mt-2 inline-flex rounded-full px-2 py-0.5 text-[10px] font-semibold uppercase tracking-wide ${badgeStyle}`}>
        {nodeType}
      </span>
    </div>
  );
}