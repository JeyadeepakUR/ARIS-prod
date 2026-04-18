import { apiFetch } from "./client";

export type DocumentItem = {
  id: string;
  workspace_id: string;
  filename: string;
  file_format: string;
  status: string;
  error_msg: string | null;
  created_at: string;
};

type UploadInitResponse = {
  document: DocumentItem;
  job_id: string;
  upload_url: string;
  status: string;
};

export async function initializeUpload(
  workspaceId: string,
  file: File,
): Promise<{ document_id: string; job_id: string; upload_url: string }> {
  const ext = file.name.split(".").pop()?.toLowerCase() ?? "txt";
  const format = ext === "pdf" ? "pdf" : "text";

  const data = await apiFetch<UploadInitResponse>(
    `/workspaces/${workspaceId}/documents/upload-url`,
    {
      method: "POST",
      body: JSON.stringify({
        filename: file.name,
        file_format: format,
        file_size_bytes: file.size,
      }),
    },
  );

  return {
    document_id: data.document.id,
    job_id: data.job_id,
    upload_url: data.upload_url,
  };
}

export async function uploadToSignedUrl(url: string, file: File): Promise<void> {
  const response = await fetch(url, { method: "PUT", body: file });
  if (!response.ok) throw new Error(`Upload failed: ${response.statusText}`);
}

export async function listDocuments(workspaceId: string): Promise<DocumentItem[]> {
  return apiFetch<DocumentItem[]>(`/workspaces/${workspaceId}/documents`);
}
