"use client";

import type { EdgeProps } from "reactflow";
import { BaseEdge, EdgeLabelRenderer, getStraightPath } from "reactflow";

import { useGraphStore } from "../../lib/stores/graphStore";

const PALETTE = [
  "#0f766e",
  "#0369a1",
  "#1d4ed8",
  "#7c3aed",
  "#b45309",
  "#be123c",
  "#15803d",
  "#374151",
];

function colorForCluster(clusterId: string): string {
  let hash = 0;
  for (let i = 0; i < clusterId.length; i += 1) {
    hash = (hash << 5) - hash + clusterId.charCodeAt(i);
    hash |= 0;
  }
  return PALETTE[Math.abs(hash) % PALETTE.length];
}

export function BridgeEdge(props: EdgeProps) {
  const bridgeFocusMode = useGraphStore((state) => state.bridgeFocusMode);
  const setActiveBridge = useGraphStore((state) => state.setActiveBridge);

  const sourceCluster = String(props.data?.source_cluster_id ?? "General Research");
  const targetCluster = String(props.data?.target_cluster_id ?? "General Research");
  const sourceColor = colorForCluster(sourceCluster);
  const targetColor = colorForCluster(targetCluster);

  const [path, labelX, labelY] = getStraightPath(props);
  const gradientId = `bridge-gradient-${props.id}`;

  const evidenceTextRaw = String(props.data?.evidence ?? "");
  const evidencePreview = evidenceTextRaw.length > 120 ? `${evidenceTextRaw.slice(0, 120)}...` : evidenceTextRaw;
  const confidence = typeof props.data?.confidence === "number" ? Number(props.data.confidence) : 0;

  const lineStyle = {
    stroke: `url(#${gradientId})`,
    strokeWidth: 2.5,
    opacity: props.style?.opacity ?? 1,
    strokeDasharray: bridgeFocusMode ? "6 3" : undefined,
  } as const;

  return (
    <>
      <defs>
        <linearGradient id={gradientId} gradientUnits="userSpaceOnUse" x1={props.sourceX} y1={props.sourceY} x2={props.targetX} y2={props.targetY}>
          <stop offset="0%" stopColor={sourceColor} />
          <stop offset="100%" stopColor={targetColor} />
        </linearGradient>
      </defs>

      <BaseEdge id={props.id} path={path} style={lineStyle} />

      <EdgeLabelRenderer>
        <button
          type="button"
          className="nodrag nopan rounded-full border border-bridge/30 bg-bridge/15 px-2.5 py-0.5 text-[10px] font-semibold text-orange-300 shadow-lg backdrop-blur"
          style={{
            position: "absolute",
            transform: `translate(-50%, -50%) translate(${labelX}px, ${labelY}px)`,
            pointerEvents: "all",
          }}
          title={`Confidence ${(confidence * 100).toFixed(0)}%\n${evidencePreview}`}
          onClick={() => setActiveBridge(props.id)}
        >
          {String(props.data?.bridge_concept ?? "Bridge")}
        </button>
      </EdgeLabelRenderer>
    </>
  );
}
