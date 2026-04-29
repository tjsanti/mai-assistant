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

export async function ingestDocuments(): Promise<{ status: string; documents_indexed: number; chunks_indexed: number }> {
  const response = await fetch(`${API_BASE_URL}/ingest`, { method: "POST" });
  if (!response.ok) {
    throw new Error("Ingestion failed.");
  }
  return response.json();
}

