from typing import Any

from fastapi.testclient import TestClient

from app.config import Settings
from app.main import app
from app.runtime import AppRuntime, get_runtime
from app.schemas import ChatRequest, ChatResponse, IngestResponse, SourceMetadata


class FakeChatAgent:
    def __init__(self) -> None:
        self.requests: list[ChatRequest] = []

    def run(self, request: ChatRequest) -> ChatResponse:
        self.requests.append(request)
        return ChatResponse(
            answer="fake answer",
            tool_used="rag_answer",
            llm_provider="ollama",
            sources=[
                SourceMetadata(
                    file="note.md",
                    page=None,
                    chunk_id="note_001",
                    score=0.9,
                )
            ],
        )


class FakeIngestionService:
    def __init__(self) -> None:
        self.calls = 0

    def run(self) -> IngestResponse:
        self.calls += 1
        return IngestResponse(
            documents_indexed=2,
            chunks_indexed=3,
        )


class FakeRuntime:
    def __init__(self) -> None:
        self.agent = FakeChatAgent()
        self.ingestion = FakeIngestionService()

    def chat_agent(self) -> FakeChatAgent:
        return self.agent

    def ingestion_service(self) -> FakeIngestionService:
        return self.ingestion


def test_chat_route_uses_runtime_dependency() -> None:
    runtime = FakeRuntime()
    app.dependency_overrides[get_runtime] = lambda: runtime
    try:
        response = TestClient(app).post("/chat", json={"message": "Use docs"})
    finally:
        app.dependency_overrides.clear()

    assert response.status_code == 200
    assert response.json()["answer"] == "fake answer"
    assert runtime.agent.requests == [ChatRequest(message="Use docs")]


def test_ingest_route_uses_runtime_dependency() -> None:
    runtime = FakeRuntime()
    app.dependency_overrides[get_runtime] = lambda: runtime
    try:
        response = TestClient(app).post("/ingest")
    finally:
        app.dependency_overrides.clear()

    assert response.status_code == 200
    assert response.json() == {
        "status": "ok",
        "documents_indexed": 2,
        "chunks_indexed": 3,
    }
    assert runtime.ingestion.calls == 1


def test_runtime_caches_composed_services(monkeypatch: Any) -> None:
    class FakeProvider:
        provider_name = "openai"

        def generate(self, messages: list[dict], **kwargs: object) -> str:
            return "answer"

        def choose_tool(self, message: str, tools: list[str]) -> str:
            return "general_response"

    class FakeRegistry:
        def __init__(self, settings: Settings) -> None:
            self.provider = FakeProvider()

        def get_tool_selection_provider(self) -> FakeProvider:
            return self.provider

        def get_rag_provider(self) -> FakeProvider:
            return self.provider

        def get_general_provider(self) -> FakeProvider:
            return self.provider

    class FakeVectorStore:
        def __init__(
            self,
            persist_directory: object,
            embedding_function: object,
        ) -> None:
            self.persist_directory = persist_directory
            self.embedding_function = embedding_function

    monkeypatch.setattr("app.runtime.ProviderRegistry", FakeRegistry)
    monkeypatch.setattr("app.runtime.get_embeddings", lambda model_name: object())
    monkeypatch.setattr("app.runtime.VectorStoreClient", FakeVectorStore)

    runtime = AppRuntime(Settings())

    assert runtime.chat_agent() is runtime.chat_agent()
    assert runtime.ingestion_service() is runtime.ingestion_service()
    assert runtime.retriever.vector_store is runtime.vector_store
