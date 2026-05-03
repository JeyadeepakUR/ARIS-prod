"use client";

import { useState } from "react";

import { DOMAIN_COLORS } from "../../lib/graph/layout";

type Props = {
  activeDomains: string[];
};

export function GraphLegend({ activeDomains }: Props) {
  const [expanded, setExpanded] = useState(false);

  // Show only domains actually present in this graph; sort by palette order.
  const sorted = activeDomains.filter((d) => DOMAIN_COLORS[d]).sort();

  return (
    <div className="absolute bottom-4 left-4 z-20 max-w-[240px] rounded-xl border border-white/10 bg-[#11131a]/85 p-3 shadow-lg backdrop-blur">
      <button
        type="button"
        onClick={() => setExpanded((v) => !v)}
        className="mb-2 flex w-full items-center justify-between text-[10px] font-semibold uppercase tracking-[0.18em] text-ink-2 transition hover:text-ink"
      >
        Legend
        <svg
          className={`h-3 w-3 transition-transform ${expanded ? "rotate-180" : ""}`}
          viewBox="0 0 24 24"
          fill="none"
          stroke="currentColor"
          strokeWidth={2}
        >
          <path strokeLinecap="round" strokeLinejoin="round" d="M19 9l-7 7-7-7" />
        </svg>
      </button>

      {/* Edge types — always visible */}
      <ul className="space-y-1.5 text-[10px] text-ink-2">
        <li className="flex items-center gap-2">
          <svg className="h-2.5 w-7 flex-shrink-0" viewBox="0 0 24 4">
            <line x1="0" y1="2" x2="24" y2="2" stroke="currentColor" strokeWidth="1.6" opacity="0.7" />
          </svg>
          <span>domain holds concept</span>
        </li>
        <li className="flex items-center gap-2">
          <svg className="h-2.5 w-7 flex-shrink-0" viewBox="0 0 24 4">
            <line
              x1="0"
              y1="2"
              x2="24"
              y2="2"
              stroke="#94a3b8"
              strokeWidth="1.2"
              strokeDasharray="3 3"
            />
          </svg>
          <span>extracted from paper</span>
        </li>
        <li className="flex items-center gap-2">
          <svg className="h-2.5 w-7 flex-shrink-0" viewBox="0 0 24 4">
            <line x1="0" y1="2" x2="24" y2="2" stroke="#f59e0b" strokeWidth="2.4" />
          </svg>
          <span className="text-amber-200">cross-domain bridge</span>
        </li>
      </ul>

      {/* Domain colours — collapsible */}
      {expanded && sorted.length > 0 && (
        <>
          <p className="mb-1.5 mt-3 text-[9px] font-semibold uppercase tracking-[0.18em] text-ink-3">
            Domains in this graph
          </p>
          <ul className="grid grid-cols-1 gap-1 text-[10px] text-ink-2">
            {sorted.map((domain) => (
              <li key={domain} className="flex items-center gap-2">
                <span
                  className="h-2 w-2 flex-shrink-0 rounded-full"
                  style={{ background: DOMAIN_COLORS[domain] }}
                />
                <span className="truncate">{domain}</span>
              </li>
            ))}
          </ul>
        </>
      )}
    </div>
  );
}
