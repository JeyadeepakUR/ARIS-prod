import { apiFetch } from "./client";

export type Workspace = {
  id: string;
  name: string;
  slug: string;
  created_at: string;
};

export async function listWorkspaces(): Promise<Workspace[]> {
  return apiFetch<Workspace[]>("/workspaces");
}

export async function createWorkspace(name: string, slug: string): Promise<Workspace> {
  return apiFetch<Workspace>("/workspaces", {
    method: "POST",
    body: JSON.stringify({ name, slug }),
  });
}
