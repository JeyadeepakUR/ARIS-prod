"use client";

import type { NodeProps } from "reactflow";
import { Handle, Position } from "reactflow";

import { colorForDomain } from "../../lib/graph/layout";
import { useGraphStore } from "../../lib/stores/graphStore";

export function DomainNode({ data }: NodeProps) {
  const label = String(data?.label ?? "Domain");
  const clusterId = String(data?.cluster_id ?? label);
  const conceptCount = Number(data?.concept_count ?? 0);
  const toggleClusterCollapse = useGraphStore((state) => state.toggleClusterCollapse);
  const color = colorForDomain(clusterId);

  return (
    <button
      type="button"
      onClick={() => toggleClusterCollapse(clusterId)}
      className="flex items-center gap-3 rounded-2xl border-2 px-4 py-2.5 text-left shadow-lg transition hover:scale-[1.02]"
      style={{
        background: `linear-gradient(135deg, ${color}26, ${color}10)`,
        borderColor: `${color}99`,
        color: "#f3f4f6",
        boxShadow: `0 4px 18px ${color}25`,
      }}
      title={`${label} — click to collapse/expand`}
    >
      <Handle
        type="target"
        position={Position.Left}
        style={{ background: color, border: "none", width: 8, height: 8 }}
      />
      <span
        className="h-2.5 w-2.5 flex-shrink-0 rounded-full"
        style={{ background: color, boxShadow: `0 0 12px ${color}` }}
      />
      <span className="flex flex-col">
        <span className="text-sm font-bold leading-tight">{label}</span>
        {conceptCount > 0 && (
          <span className="text-[10px] uppercase tracking-wider opacity-70">
            {conceptCount} concept{conceptCount === 1 ? "" : "s"}
          </span>
        )}
      </span>
      <Handle
        type="source"
        position={Position.Right}
        style={{ background: color, border: "none", width: 8, height: 8 }}
      />
    </button>
  );
}
