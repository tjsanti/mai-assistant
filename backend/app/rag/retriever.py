from dataclasses import dataclass

from langchain_core.documents import Document


@dataclass(slots=True)
class RetrievedChunk:
    document: Document
    score: float


class Retriever:
    def __init__(self, vector_store: object, top_k: int) -> None:
        self.vector_store = vector_store
        self.top_k = top_k

    def retrieve(self, query: str) -> list[RetrievedChunk]:
        results = self.vector_store.similarity_search(query, k=self.top_k)
        return [RetrievedChunk(document=doc, score=score) for doc, score in results]

