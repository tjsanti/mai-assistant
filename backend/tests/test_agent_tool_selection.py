from app.agent.graph import AgentGraph
from app.agent.tools import GeneralResponseTool
from app.rag.rag_answer_tool import RagAnswerTool
from app.schemas import ChatRequest, RagAnswerResult, SourceMetadata


class FakeSelector:
    def __init__(self, tool_name: str) -> None:
        self.tool_name = tool_name

    def choose_tool(self, message: str, tools: list[str]) -> str:
        return self.tool_name


class FakeRagTool:
    def run(self, query: str) -> RagAnswerResult:
        return RagAnswerResult(
            answer="rag answer",
            llm_provider="ollama",
            sources=[SourceMetadata(file="resume.md", page=None, chunk_id="resume_001", score=0.9)],
        )


class FakeProvider:
    provider_name = "openai"

    def generate(self, messages: list[dict], **kwargs: object) -> str:
        return "general answer"


def test_calls_rag_answer_for_document_question() -> None:
    graph = AgentGraph(FakeSelector("rag_answer"), FakeRagTool(), GeneralResponseTool(FakeProvider()))
    response = graph.run(ChatRequest(message="What does my resume say about FAISS?"))
    assert response.tool_used == "rag_answer"
    assert response.llm_provider == "ollama"


def test_calls_general_response_for_general_question() -> None:
    graph = AgentGraph(
        FakeSelector("general_response"),
        FakeRagTool(),
        GeneralResponseTool(FakeProvider()),
    )
    response = graph.run(ChatRequest(message="What is the capital of France?"))
    assert response.tool_used == "general_response"
    assert response.llm_provider == "openai"


def test_respects_force_tool() -> None:
    graph = AgentGraph(
        FakeSelector("general_response"),
        FakeRagTool(),
        GeneralResponseTool(FakeProvider()),
    )
    response = graph.run(ChatRequest(message="Use docs", force_tool="rag_answer"))
    assert response.tool_used == "rag_answer"


def test_returns_tool_and_provider_metadata() -> None:
    graph = AgentGraph(FakeSelector("rag_answer"), FakeRagTool(), GeneralResponseTool(FakeProvider()))
    response = graph.run(ChatRequest(message="Use docs"))
    assert response.tool_used == "rag_answer"
    assert response.sources[0].file == "resume.md"

