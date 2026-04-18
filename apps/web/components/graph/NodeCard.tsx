import { Handle, Position } from "reactflow";

type NodeCardProps = {
  label: string;
  nodeType: string;
};

const TYPE_STYLE: Record<string, { border: string; bg: string; badgeCls: string }> = {
  document: { border: "rgba(99,102,241,0.3)", bg: "rgba(99,102,241,0.08)", badgeCls: "bg-accent/15 text-accent-2" },
  domain: { border: "rgba(20,184,166,0.35)", bg: "rgba(20,184,166,0.08)", badgeCls: "bg-success/15 text-emerald-300" },
  concept: { border: "rgba(255,255,255,0.1)", bg: "rgba(255,255,255,0.04)", badgeCls: "bg-white/8 text-ink-2" },
  bridge_concept: { border: "rgba(249,115,22,0.35)", bg: "rgba(249,115,22,0.08)", badgeCls: "bg-bridge/15 text-orange-300" },
};

export function NodeCard({ label, nodeType }: NodeCardProps) {
  const s = TYPE_STYLE[nodeType] ?? TYPE_STYLE.concept;

  return (
    <div
      className="min-w-[160px] rounded-xl p-3 shadow-lg"
      style={{ border: `1px solid ${s.border}`, background: s.bg }}
    >
      <Handle type="target" position={Position.Left} style={{ background: s.border, border: "none", width: 7, height: 7 }} />
      <Handle type="source" position={Position.Right} style={{ background: s.border, border: "none", width: 7, height: 7 }} />
      <p className="line-clamp-2 text-[13px] font-semibold leading-snug text-ink">{label}</p>
      <span className={`mt-2 inline-flex rounded-full px-2 py-0.5 text-[10px] font-semibold uppercase tracking-wider ${s.badgeCls}`}>
        {nodeType.replace("_", " ")}
      </span>
    </div>
  );
}