from langchain_core.documents import Document

from app.rag.retriever import Retriever


class FakeVectorStore:
    def __init__(self, results: list[tuple[Document, float]]) -> None:
        self.results = results

    def similarity_search(self, query: str, k: int) -> list[tuple[Document, float]]:
        return self.results[:k]


def test_returns_top_k_chunks() -> None:
    documents = [
        (Document(page_content="a", metadata={"file": "a.txt", "chunk_id": "a_001"}), 0.9),
        (Document(page_content="b", metadata={"file": "b.txt", "chunk_id": "b_001"}), 0.8),
    ]
    retriever = Retriever(FakeVectorStore(documents), top_k=1)
    results = retriever.retrieve("query")
    assert len(results) == 1


def test_includes_file_metadata() -> None:
    retriever = Retriever(
        FakeVectorStore(
            [(Document(page_content="a", metadata={"file": "a.txt", "chunk_id": "a_001"}), 0.9)]
        ),
        top_k=5,
    )
    results = retriever.retrieve("query")
    assert results[0].metadata.file == "a.txt"


def test_handles_empty_vector_store_gracefully() -> None:
    retriever = Retriever(FakeVectorStore([]), top_k=5)
    assert retriever.retrieve("query") == []
