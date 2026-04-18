import { apiFetch } from "./client";

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
  return apiFetch<GraphNode[]>(`/graphs/${graphId}/nodes`);
}

export async function listGraphEdges(graphId: string): Promise<GraphEdge[]> {
  return apiFetch<GraphEdge[]>(`/graphs/${graphId}/edges`);
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
