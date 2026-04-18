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
      className="flex flex-col items-center gap-1 group"
    >
      <span
        className="h-7 w-7 rotate-45 rounded-md shadow-lg transition group-hover:scale-110"
        style={{ background: "rgba(249,115,22,0.9)", border: "1.5px solid rgba(249,115,22,0.4)" }}
        aria-hidden
      />
      <span className="max-w-[140px] rounded-full border border-bridge/30 bg-bridge/10 px-2 py-0.5 text-center text-[10px] font-semibold text-orange-300">
        {bridgeConcept}
      </span>
    </button>
  );
}
