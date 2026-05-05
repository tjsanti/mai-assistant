from app.rag.chunks import ChunkMetadata
from app.rag.rag_answerer import UNSUPPORTED_ANSWER, RagAnswerer
from app.rag.retriever import RetrievedChunk


class FakeRetriever:
    def __init__(self, results: list[RetrievedChunk]) -> None:
        self.results = results
        self.calls: list[str] = []

    def retrieve(self, query: str) -> list[RetrievedChunk]:
        self.calls.append(query)
        return self.results


class FakeProvider:
    provider_name = "ollama"

    def __init__(self, answer: str) -> None:
        self.answer = answer
        self.calls = 0

    def generate(self, messages: list[dict], **kwargs: object) -> str:
        self.calls += 1
        return self.answer


def test_calls_retriever_and_uses_local_provider() -> None:
    retriever = FakeRetriever(
        [
            RetrievedChunk(
                content="FAISS experience",
                metadata=ChunkMetadata(
                    file="resume.md",
                    path=None,
                    chunk_id="resume_001",
                    page=None,
                ),
                score=0.82,
            )
        ]
    )
    provider = FakeProvider("You mention FAISS in resume.md.")
    result = RagAnswerer(retriever, provider).run("What does my resume say about FAISS?")
    assert retriever.calls == ["What does my resume say about FAISS?"]
    assert provider.calls == 1
    assert result.llm_provider == "ollama"


def test_includes_sources() -> None:
    retriever = FakeRetriever(
        [
            RetrievedChunk(
                content="content",
                metadata=ChunkMetadata(
                    file="resume.md",
                    path=None,
                    chunk_id="resume_001",
                    page=2,
                ),
                score=0.82,
            )
        ]
    )
    result = RagAnswerer(retriever, FakeProvider("answer")).run("query")
    assert result.sources[0].file == "resume.md"
    assert result.sources[0].page == 2


def test_refuses_unsupported_answers() -> None:
    result = RagAnswerer(FakeRetriever([]), FakeProvider("should not be used")).run("query")
    assert result.answer == UNSUPPORTED_ANSWER


def test_does_not_call_openai_by_default() -> None:
    provider = FakeProvider("answer")
    RagAnswerer(FakeRetriever([]), provider).run("query")
    assert provider.calls == 0
