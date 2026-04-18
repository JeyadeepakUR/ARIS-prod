"use client";

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
  const isBridge = bridgeFocusMode || bridgeFocus;

  return (
    <div className="flex items-center gap-3">
      <button
        type="button"
        onClick={onFit}
        title="Fit graph to view"
        className="flex h-8 w-8 items-center justify-center rounded-lg border border-white/10 bg-white/5 text-ink-2 transition hover:bg-white/10 hover:text-ink"
      >
        <svg className="h-3.5 w-3.5" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}>
          <path strokeLinecap="round" strokeLinejoin="round" d="M4 8V4h4M20 8V4h-4M4 16v4h4M20 16v4h-4"/>
        </svg>
      </button>

      <label className="flex items-center gap-2 text-[11px] text-ink-2">
        <span className="whitespace-nowrap">≥{(confidenceThreshold * 100).toFixed(0)}%</span>
        <input
          type="range"
          min={0}
          max={1}
          step={0.05}
          value={confidenceThreshold}
          onChange={(event) => onChangeThreshold(Number(event.target.value))}
          className="w-24"
        />
      </label>

      <button
        type="button"
        onClick={() => { toggleBridgeFocus(); onToggleBridgeFocus(); }}
        className={`flex items-center gap-1.5 rounded-lg px-3 py-1.5 text-[11px] font-semibold transition ${
          isBridge
            ? "bg-bridge/20 border border-bridge/30 text-orange-300"
            : "border border-white/10 bg-white/5 text-ink-2 hover:bg-white/10"
        }`}
      >
        <span className={`h-1.5 w-1.5 rounded-full ${isBridge ? "bg-bridge" : "bg-ink-3"}`} />
        Bridges {isBridge ? "on" : "off"}
      </button>
    </div>
  );
}
