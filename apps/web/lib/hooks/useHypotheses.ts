"use client";

import { useCallback, useEffect, useState } from "react";
import {
  listHypotheses,
  updateHypothesisStatus,
  type Hypothesis,
  type HypothesisStatus,
} from "../api/graphs";

export function useHypotheses(workspaceId: string | null, graphId: string | null) {
  const [hypotheses, setHypotheses] = useState<Hypothesis[]>([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    if (!workspaceId || !graphId) return;
    let mounted = true;
    setLoading(true);
    listHypotheses(workspaceId, graphId)
      .then((items) => { if (mounted) setHypotheses(items); })
      .catch((err) => { if (mounted) setError(err instanceof Error ? err.message : "Failed to load hypotheses"); })
      .finally(() => { if (mounted) setLoading(false); });
    return () => { mounted = false; };
  }, [workspaceId, graphId]);

  const updateStatus = useCallback(
    async (hypothesisId: string, status: HypothesisStatus) => {
      if (!workspaceId || !graphId) return;
      const updated = await updateHypothesisStatus(workspaceId, graphId, hypothesisId, status);
      setHypotheses((prev) => prev.map((h) => (h.id === hypothesisId ? updated : h)));
    },
    [workspaceId, graphId],
  );

  return { hypotheses, loading, isLoading: loading, error, updateStatus };
}
