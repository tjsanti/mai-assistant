from typing import Literal

from pydantic import BaseModel, Field


ToolName = Literal["rag_answer", "general_response"]
ProviderName = Literal["openai", "ollama"]


class SourceMetadata(BaseModel):
    file: str
    page: int | None = None
    chunk_id: str
    score: float | None = None


class ChatRequest(BaseModel):
    message: str
    force_tool: ToolName | None = None


class ChatResponse(BaseModel):
    answer: str
    tool_used: ToolName
    llm_provider: ProviderName
    sources: list[SourceMetadata] = Field(default_factory=list)


class HealthResponse(BaseModel):
    status: Literal["ok"] = "ok"


class IngestResponse(BaseModel):
    status: Literal["ok"] = "ok"
    documents_indexed: int
    chunks_indexed: int
