"use client";

import { useCallback, useEffect, useRef, useState } from "react";
import { STREAM_BASE } from "../api/graphs";

export type StreamEvent = {
  id?: string;
  event_type: string;
  agent_node?: string;
  content?: Record<string, unknown>;
  status?: string;
  graph_id?: string;
  timestamp?: string;
  message?: string;
};

export type StreamStatus = "idle" | "connecting" | "streaming" | "complete" | "error";

export function useGraphStream(graphId: string | null, enabled: boolean) {
  const [events, setEvents] = useState<StreamEvent[]>([]);
  const [status, setStatus] = useState<StreamStatus>("idle");
  const [error, setError] = useState<string | null>(null);
  const abortRef = useRef<AbortController | null>(null);

  const stop = useCallback(() => {
    abortRef.current?.abort();
    abortRef.current = null;
  }, []);

  const reset = useCallback(() => {
    stop();
    setEvents([]);
    setStatus("idle");
    setError(null);
  }, [stop]);

  useEffect(() => {
    if (!enabled || !graphId) return;

    const token =
      typeof window !== "undefined" ? localStorage.getItem("access_token") : null;

    const controller = new AbortController();
    abortRef.current = controller;

    setStatus("connecting");
    setEvents([]);
    setError(null);

    (async () => {
      try {
        const response = await fetch(`${STREAM_BASE}/graphs/${graphId}/stream`, {
          headers: token ? { Authorization: `Bearer ${token}` } : {},
          signal: controller.signal,
        });

        if (!response.ok) {
          throw new Error(`Stream HTTP ${response.status}`);
        }

        setStatus("streaming");

        const reader = response.body!.getReader();
        const decoder = new TextDecoder();
        let buffer = "";

        while (true) {
          const { done, value } = await reader.read();
          if (done) break;

          buffer += decoder.decode(value, { stream: true });
          const lines = buffer.split("\n");
          buffer = lines.pop() ?? "";

          for (const line of lines) {
            const trimmed = line.trim();
            if (!trimmed.startsWith("data:")) continue;
            const payload = trimmed.slice(5).trim();
            if (!payload) continue;
            try {
              const evt = JSON.parse(payload) as StreamEvent;
              setEvents((prev) => [...prev, evt]);
              if (evt.event_type === "run_complete" || evt.event_type === "error") {
                setStatus("complete");
                controller.abort();
                return;
              }
            } catch {
              // malformed line — skip
            }
          }
        }

        setStatus("complete");
      } catch (err) {
        if ((err as Error).name === "AbortError") return;
        setError((err as Error).message);
        setStatus("error");
      }
    })();

    return () => {
      controller.abort();
    };
  }, [graphId, enabled]);

  return { events, status, error, reset };
}
