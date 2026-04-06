"use client";

import type { NodeProps } from "reactflow";

import { useGraphStore } from "../../lib/stores/graphStore";

export function BridgeMarkerNode({ data }: NodeProps) {
  const setActiveBridge = useGraphStore((state) => state.setActiveBridge);
  const bridgeConcept = String(data?.bridge_concept ?? "bridge");
  const edgeId = String(data?.edge_id ?? "");

  return (
    <button
      type="button"
      onClick={() => setActiveBridge(edgeId || null)}
      className="flex flex-col items-center gap-1"
    >
      <span
        className="h-8 w-8 rotate-45 rounded-[4px] bg-slate-800"
        aria-hidden
      />
      <span className="max-w-[180px] text-center text-[10px] italic text-slate-700">{bridgeConcept}</span>
    </button>
  );
}
