from functools import cached_property, lru_cache

from app.agent.answer_engine import AnswerEngine
from app.agent.general_answerer import GeneralAnswerer
from app.agent.graph import AgentGraph
from app.config import Settings, get_settings
from app.llms.router import ProviderRegistry
from app.rag.embeddings import get_embeddings
from app.rag.ingestion import IngestionService
from app.rag.rag_answerer import RagAnswerer
from app.rag.retriever import Retriever
from app.rag.vector_store import VectorStoreClient


class AppRuntime:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings

    @cached_property
    def provider_registry(self) -> ProviderRegistry:
        return ProviderRegistry(self.settings)

    @cached_property
    def vector_store(self) -> VectorStoreClient:
        embeddings = get_embeddings(self.settings.embedding_model)
        return VectorStoreClient(self.settings.vector_db_dir, embeddings)

    @cached_property
    def retriever(self) -> Retriever:
        return Retriever(
            vector_store=self.vector_store,
            top_k=self.settings.retrieval_top_k,
        )

    @cached_property
    def answer_engine(self) -> AnswerEngine:
        return AnswerEngine(
            tool_selector=self.provider_registry.get_tool_selection_provider(),
            rag_answerer=RagAnswerer(
                retriever=self.retriever,
                provider=self.provider_registry.get_rag_provider(),
            ),
            general_answerer=GeneralAnswerer(
                self.provider_registry.get_general_provider(),
            ),
        )

    @cached_property
    def _chat_agent(self) -> AgentGraph:
        return AgentGraph(answer_engine=self.answer_engine)

    @cached_property
    def _ingestion_service(self) -> IngestionService:
        return IngestionService(
            docs_dir=self.settings.docs_dir,
            vector_store=self.vector_store,
        )

    def chat_agent(self) -> AgentGraph:
        return self._chat_agent

    def ingestion_service(self) -> IngestionService:
        return self._ingestion_service


@lru_cache
def get_runtime() -> AppRuntime:
    return AppRuntime(get_settings())
