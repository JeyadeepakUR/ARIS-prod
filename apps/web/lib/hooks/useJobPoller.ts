"use client";

import { useEffect, useRef, useState } from "react";
import { getJob, type Job } from "../api/jobs";

const TERMINAL_STATUSES = new Set(["ready", "failed", "complete", "completed"]);
const POLL_INTERVAL_MS = 2000;

export function useJobPoller(jobId: string | null, enabled: boolean) {
  const [job, setJob] = useState<Job | null>(null);
  const [error, setError] = useState<string | null>(null);
  const intervalRef = useRef<ReturnType<typeof setInterval> | null>(null);

  const isComplete = job ? TERMINAL_STATUSES.has(job.status) : false;

  useEffect(() => {
    if (!enabled || !jobId) {
      setJob(null);
      setError(null);
      return;
    }

    let active = true;

    async function poll() {
      if (!jobId) return;
      try {
        const result = await getJob(jobId);
        if (!active) return;
        setJob(result);
        setError(null);
        if (TERMINAL_STATUSES.has(result.status) && intervalRef.current) {
          clearInterval(intervalRef.current);
          intervalRef.current = null;
        }
      } catch (err) {
        if (!active) return;
        setError(err instanceof Error ? err.message : "Polling failed");
      }
    }

    poll();
    intervalRef.current = setInterval(poll, POLL_INTERVAL_MS);

    return () => {
      active = false;
      if (intervalRef.current) {
        clearInterval(intervalRef.current);
        intervalRef.current = null;
      }
    };
  }, [jobId, enabled]);

  return { job, error, isComplete };
}
