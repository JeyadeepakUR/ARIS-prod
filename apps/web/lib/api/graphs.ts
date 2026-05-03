import { apiFetch } from "./client";

export const STREAM_BASE = process.env.NEXT_PUBLIC_API_URL ?? "http://127.0.0.1:8000";

export type Graph = {
  id: string;
  workspace_id: string;
  status: string;
  metadata: Record<string, unknown>;
  created_at: string;
};

export type GraphNode = {
  id: string;
  graph_id: string;
  document_id: string | null;
  label: string;
  node_type: string;
  tier: number;
  cluster_id: string | null;
  metadata: Record<string, unknown>;
  created_at: string;
};

export type GraphEdge = {
  id: string;
  graph_id: string;
  source_node_id: string;
  target_node_id: string;
  edge_type: string;
  edge_category: string;
  bridge_concept: string | null;
  confidence: number;
  evidence: { text: string; reasoning_trace_id: string; confidence: number };
  metadata: Record<string, unknown>;
  created_at: string;
};

export type HypothesisStatus = "proposed" | "investigating" | "accepted" | "rejected";

export type Hypothesis = {
  id: string;
  graph_id: string;
  edge_id: string;
  hypothesis_text: string;
  confidence: number;
  status: HypothesisStatus;
  created_at: string;
};

export async function listGraphs(workspaceId: string): Promise<Graph[]> {
  return apiFetch<Graph[]>(`/workspaces/${workspaceId}/graphs`);
}

export async function getGraph(workspaceId: string, graphId: string): Promise<Graph> {
  return apiFetch<Graph>(`/workspaces/${workspaceId}/graphs/${graphId}`);
}

export async function buildGraph(
  workspaceId: string,
  documentIds: string[],
  strategy = "domain_network",
): Promise<{ graph: Graph; job_id: string; status: string }> {
  return apiFetch(`/workspaces/${workspaceId}/graphs`, {
    method: "POST",
    body: JSON.stringify({
      document_ids: documentIds,
      strategy,
      plan_strategy: "weak-evidence",
      max_plan_actions: 10,
    }),
  });
}

export async function listGraphNodes(graphId: string): Promise<GraphNode[]> {
  return apiFetch<GraphNode[]>(`/graphs/${graphId}/nodes?size=500`);
}

export async function listGraphEdges(graphId: string): Promise<GraphEdge[]> {
  return apiFetch<GraphEdge[]>(`/graphs/${graphId}/edges?size=500`);
}

export async function listHypotheses(
  workspaceId: string,
  graphId: string,
): Promise<Hypothesis[]> {
  return apiFetch<Hypothesis[]>(
    `/workspaces/${workspaceId}/graphs/${graphId}/hypotheses`,
  );
}

export async function updateHypothesisStatus(
  workspaceId: string,
  graphId: string,
  hypothesisId: string,
  status: HypothesisStatus,
): Promise<Hypothesis> {
  return apiFetch<Hypothesis>(
    `/workspaces/${workspaceId}/graphs/${graphId}/hypotheses/${hypothesisId}`,
    { method: "PATCH", body: JSON.stringify({ status }) },
  );
}

export type Contradiction = {
  id: string;
  graph_id: string;
  claim_a_text: string;
  claim_b_text: string;
  author_a: string | null;
  author_b: string | null;
  contradiction_type: string;
  severity: number;
  llm_reasoning: string | null;
  status: string;
  claim_a_chunk_id?: string | null;
  claim_b_chunk_id?: string | null;
  created_at: string;
};

export type PlanAction = {
  id: string;
  graph_id: string;
  action_type: string;
  description: string;
  evidence: string;
  rationale: string;
  priority: number;
  status: string;
  metadata: Record<string, unknown>;
  created_at: string;
};

export async function listGraphContradictions(graphId: string): Promise<Contradiction[]> {
  return apiFetch<Contradiction[]>(`/graphs/${graphId}/contradictions`);
}

export async function listGraphPlans(graphId: string): Promise<PlanAction[]> {
  return apiFetch<PlanAction[]>(`/graphs/${graphId}/plans`);
}

export type ChunkEvidence = {
  chunk_id: string;
  document_id: string;
  document_title: string;
  document_filename: string;
  document_s3_key: string | null;
  page_number: number | null;
  section: string | null;
  chunk_index: number;
  content: string;
};

export async function getChunkEvidence(
  graphId: string,
  chunkIds: string[],
): Promise<ChunkEvidence[]> {
  const ids = chunkIds.filter(Boolean).join(",");
  if (!ids) return [];
  return apiFetch<ChunkEvidence[]>(
    `/graphs/${graphId}/chunks?ids=${encodeURIComponent(ids)}`,
  );
}

const _DEFAULT_BUCKET = "aris-documents";

export function buildDocumentDownloadUrl(s3Key: string, page?: number | null): string {
  const base = STREAM_BASE.replace(/\/$/, "");
  const url = `${base}/object-storage/${_DEFAULT_BUCKET}/${s3Key.split("/").map(encodeURIComponent).join("/")}`;
  return page ? `${url}#page=${page}` : url;
}
