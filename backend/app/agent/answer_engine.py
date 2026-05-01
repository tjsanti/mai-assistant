from typing import Protocol

from app.config import Settings
from app.llms.base import ToolSelectionProvider
from app.llms.router import ProviderRegistry
from app.rag.embeddings import get_embeddings
from app.rag.rag_answerer import RagAnswerer
from app.rag.retriever import Retriever
from app.rag.vector_store import VectorStoreClient
from app.schemas import ChatRequest, ChatResponse, ToolName


class QueryAnswerer(Protocol):
    def run(self, query: str) -> ChatResponse:
        ...


class AnswerEngine:
    def __init__(
        self,
        tool_selector: ToolSelectionProvider,
        rag_answerer: QueryAnswerer,
        general_answerer: QueryAnswerer,
    ) -> None:
        self.tool_selector = tool_selector
        self.rag_answerer = rag_answerer
        self.general_answerer = general_answerer

    @classmethod
    def from_settings(cls, settings: Settings) -> "AnswerEngine":
        registry = ProviderRegistry(settings)
        embeddings = get_embeddings(settings.embedding_model)
        vector_store = VectorStoreClient(settings.vector_db_dir, embeddings)
        retriever = Retriever(vector_store=vector_store, top_k=settings.retrieval_top_k)
        return cls(
            tool_selector=registry.get_tool_selection_provider(),
            rag_answerer=RagAnswerer(
                retriever=retriever,
                provider=registry.get_rag_provider(),
            ),
            general_answerer=registry.get_general_answerer(),
        )

    def choose_tool(self, request: ChatRequest) -> ToolName:
        if request.force_tool:
            return request.force_tool
        return self.tool_selector.choose_tool(
            message=request.message,
            tools=["rag_answer", "general_response"],
        )

    def answer(self, request: ChatRequest) -> ChatResponse:
        return self.answer_with_tool(request.message, self.choose_tool(request))

    def answer_with_tool(self, query: str, tool_name: ToolName) -> ChatResponse:
        if tool_name == "rag_answer":
            return self.rag_answerer.run(query)
        return self.general_answerer.run(query)
