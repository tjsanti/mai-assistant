export type ToolName = "rag_answer" | "general_response";

export type SourceMetadata = {
  file: string;
  page: number | null;
  chunk_id: string;
  score: number | null;
};

export type ChatResponse = {
  answer: string;
  tool_used: ToolName;
  llm_provider: "openai" | "ollama";
  sources: SourceMetadata[];
};

export type IngestResponse = {
  status: string;
  documents_indexed: number;
  chunks_indexed: number;
};

export type UploadDocumentsResponse = IngestResponse & {
  uploaded_files: string[];
};

const API_BASE_URL = "http://localhost:8000";

export async function sendChat(message: string, forceTool: ToolName | null = null): Promise<ChatResponse> {
  const response = await fetch(`${API_BASE_URL}/chat`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ message, force_tool: forceTool }),
  });

  if (!response.ok) {
    throw new Error("Chat request failed.");
  }

  return response.json();
}

export async function ingestDocuments(): Promise<IngestResponse> {
  const response = await fetch(`${API_BASE_URL}/ingest`, { method: "POST" });
  if (!response.ok) {
    throw new Error("Ingestion failed.");
  }
  return response.json();
}

export async function uploadDocuments(files: File[]): Promise<UploadDocumentsResponse> {
  const formData = new FormData();
  files.forEach((file) => formData.append("files", file));

  const response = await fetch(`${API_BASE_URL}/documents/upload`, {
    method: "POST",
    body: formData,
  });

  if (!response.ok) {
    const detail = await response.json().catch(() => null);
    throw new Error(detail?.detail ?? "Document upload failed.");
  }

  return response.json();
}
