import { apiFetch } from "./client";

export type Job = {
  id: string;
  workspace_id: string | null;
  job_type: string;
  status: string;
  result: Record<string, unknown>;
  error: string | null;
  created_at: string;
  started_at: string | null;
  completed_at: string | null;
};

export async function getJob(jobId: string): Promise<Job> {
  return apiFetch<Job>(`/jobs/${jobId}`);
}
