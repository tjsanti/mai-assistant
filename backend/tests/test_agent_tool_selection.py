from app.agent.answer_engine import AnswerEngine
from app.agent.general_answerer import GeneralAnswerer
from app.agent.graph import AgentGraph
from app.schemas import ChatRequest, ChatResponse, SourceMetadata


class FakeSelector:
    def __init__(self, tool_name: str) -> None:
        self.tool_name = tool_name

    def choose_tool(self, message: str, tools: list[str]) -> str:
        return self.tool_name


class FakeRagTool:
    def run(self, query: str) -> ChatResponse:
        return ChatResponse(
            answer="rag answer",
            tool_used="rag_answer",
            llm_provider="ollama",
            sources=[SourceMetadata(file="resume.md", page=None, chunk_id="resume_001", score=0.9)],
        )


class FakeProvider:
    provider_name = "openai"

    def generate(self, messages: list[dict], **kwargs: object) -> str:
        return "general answer"


def test_calls_rag_answer_for_document_question() -> None:
    engine = AnswerEngine(FakeSelector("rag_answer"), FakeRagTool(), GeneralAnswerer(FakeProvider()))
    response = engine.answer(ChatRequest(message="What does my resume say about FAISS?"))
    assert response.tool_used == "rag_answer"
    assert response.llm_provider == "ollama"


def test_calls_general_response_for_general_question() -> None:
    engine = AnswerEngine(
        FakeSelector("general_response"), FakeRagTool(), GeneralAnswerer(FakeProvider())
    )
    response = engine.answer(ChatRequest(message="What is the capital of France?"))
    assert response.tool_used == "general_response"
    assert response.llm_provider == "openai"


def test_respects_force_tool() -> None:
    engine = AnswerEngine(
        FakeSelector("general_response"), FakeRagTool(), GeneralAnswerer(FakeProvider())
    )
    response = engine.answer(ChatRequest(message="Use docs", force_tool="rag_answer"))
    assert response.tool_used == "rag_answer"


def test_returns_tool_and_provider_metadata() -> None:
    engine = AnswerEngine(FakeSelector("rag_answer"), FakeRagTool(), GeneralAnswerer(FakeProvider()))
    response = engine.answer(ChatRequest(message="Use docs"))
    assert response.tool_used == "rag_answer"
    assert response.sources[0].file == "resume.md"


def test_graph_delegates_execution_to_answer_engine() -> None:
    engine = AnswerEngine(FakeSelector("general_response"), FakeRagTool(), GeneralAnswerer(FakeProvider()))
    graph = AgentGraph(engine)
    response = graph.run(ChatRequest(message="What is the capital of France?"))
    assert response.tool_used == "general_response"
