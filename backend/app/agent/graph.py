from typing import TypedDict

from langgraph.graph import END, START, StateGraph

from app.agent.tools import GeneralResponseTool
from app.config import Settings
from app.llms.router import ProviderRegistry
from app.rag.embeddings import get_embeddings
from app.rag.rag_answer_tool import RagAnswerTool
from app.rag.retriever import Retriever
from app.rag.vector_store import VectorStoreClient
from app.schemas import ChatRequest, ChatResponse


class AgentState(TypedDict, total=False):
    query: str
    force_tool: str | None
    chosen_tool: str
    response: ChatResponse


class AgentGraph:
    def __init__(
        self,
        tool_selector: object,
        rag_tool: RagAnswerTool,
        general_tool: GeneralResponseTool,
    ) -> None:
        self.tool_selector = tool_selector
        self.rag_tool = rag_tool
        self.general_tool = general_tool
        self.graph = self._build_graph()

    @classmethod
    def from_settings(cls, settings: Settings) -> "AgentGraph":
        registry = ProviderRegistry(settings)
        embeddings = get_embeddings(settings.embedding_model)
        vector_store = VectorStoreClient(settings.vector_db_dir, embeddings)
        retriever = Retriever(vector_store=vector_store, top_k=settings.retrieval_top_k)
        rag_tool = RagAnswerTool(retriever=retriever, provider=registry.get_rag_provider())
        general_tool = GeneralResponseTool(registry.get_general_provider())
        selector = registry.get_tool_selection_provider()
        return cls(tool_selector=selector, rag_tool=rag_tool, general_tool=general_tool)

    def run(self, request: ChatRequest) -> ChatResponse:
        state = self.graph.invoke({"query": request.message, "force_tool": request.force_tool})
        return state["response"]

    def _build_graph(self) -> object:
        graph = StateGraph(AgentState)
        graph.add_node("route", self._route)
        graph.add_node("rag_answer", self._run_rag)
        graph.add_node("general_response", self._run_general)
        graph.add_edge(START, "route")
        graph.add_conditional_edges(
            "route",
            lambda state: state["chosen_tool"],
            {
                "rag_answer": "rag_answer",
                "general_response": "general_response",
            },
        )
        graph.add_edge("rag_answer", END)
        graph.add_edge("general_response", END)
        return graph.compile()

    def _route(self, state: AgentState) -> AgentState:
        if state.get("force_tool"):
            return {"chosen_tool": state["force_tool"]}
        chosen = self.tool_selector.choose_tool(
            message=state["query"],
            tools=["rag_answer", "general_response"],
        )
        return {"chosen_tool": chosen}

    def _run_rag(self, state: AgentState) -> AgentState:
        result = self.rag_tool.run(state["query"])
        return {
            "response": ChatResponse(
                answer=result.answer,
                tool_used="rag_answer",
                llm_provider=result.llm_provider,
                sources=result.sources,
            )
        }

    def _run_general(self, state: AgentState) -> AgentState:
        return {"response": self.general_tool.run(state["query"])}

