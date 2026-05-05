from typing import TypedDict

from langgraph.graph import END, START, StateGraph

from app.agent.answer_engine import AnswerEngine
from app.schemas import ChatRequest, ChatResponse, ToolName


class AgentState(TypedDict, total=False):
    query: str
    force_tool: ToolName | None
    chosen_tool: ToolName
    response: ChatResponse


class AgentGraph:
    def __init__(self, answer_engine: AnswerEngine) -> None:
        self.answer_engine = answer_engine
        self.graph = self._build_graph()

    def run(self, request: ChatRequest) -> ChatResponse:
        state = self.graph.invoke(
            {"query": request.message, "force_tool": request.force_tool}
        )
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
        request = ChatRequest(
            message=state["query"],
            force_tool=state.get("force_tool"),
        )
        return {"chosen_tool": self.answer_engine.choose_tool(request)}

    def _run_rag(self, state: AgentState) -> AgentState:
        return {"response": self.answer_engine.answer_with_tool(state["query"], "rag_answer")}

    def _run_general(self, state: AgentState) -> AgentState:
        return {
            "response": self.answer_engine.answer_with_tool(
                state["query"], "general_response"
            )
        }
