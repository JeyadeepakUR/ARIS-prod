"use client";

import { useViewport } from "reactflow";

import { useGraphStore } from "../../lib/stores/graphStore";

type GraphControlsProps = {
  confidenceThreshold: number;
  onChangeThreshold: (value: number) => void;
  onFit: () => void;
  bridgeFocus: boolean;
  onToggleBridgeFocus: () => void;
};

export function GraphControls({
  confidenceThreshold,
  onChangeThreshold,
  onFit,
  bridgeFocus,
  onToggleBridgeFocus,
}: GraphControlsProps) {
  const bridgeFocusMode = useGraphStore((state) => state.bridgeFocusMode);
  const toggleBridgeFocus = useGraphStore((state) => state.toggleBridgeFocus);
  const viewport = useViewport();
  const zoomLevel = viewport?.zoom ?? 1;

  return (
    <div className="mb-4 flex flex-wrap items-center gap-4 rounded-xl border border-spice/20 bg-white/75 p-3">
      <button
        type="button"
        onClick={onFit}
        className="rounded-lg bg-pine px-3 py-2 text-xs font-semibold uppercase tracking-wide text-white"
      >
        Fit Graph
      </button>

      <label className="flex min-w-[220px] flex-1 items-center gap-3 text-xs font-semibold uppercase tracking-wide text-spice">
        Confidence {confidenceThreshold.toFixed(2)}
        <input
          type="range"
          min={0}
          max={1}
          step={0.05}
          value={confidenceThreshold}
          onChange={(event) => onChangeThreshold(Number(event.target.value))}
          className="w-full"
        />
      </label>

      <button
        type="button"
        onClick={() => {
          toggleBridgeFocus();
          onToggleBridgeFocus();
        }}
        className={`rounded-lg px-3 py-2 text-xs font-semibold uppercase tracking-wide ${
          bridgeFocusMode || bridgeFocus
            ? "bg-spice text-white"
            : "border border-spice/25 bg-white text-spice"
        }`}
      >
        {bridgeFocusMode || bridgeFocus ? "Bridge Focus On" : "Bridge Focus Off"}
      </button>

      <span className="rounded-md border border-spice/25 bg-white px-2 py-1 text-[11px] font-semibold uppercase tracking-wide text-spice">
        Zoom {zoomLevel.toFixed(2)}x
      </span>

      <p className="text-[11px] font-medium text-spice/80">Concept nodes visible at zoom &gt; 1.2x</p>
    </div>
  );
}
