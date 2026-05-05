from pathlib import Path

from langchain_chroma import Chroma
from langchain_core.documents import Document

from app.rag.chunks import chunk_id_for


class VectorStoreClient:
    def __init__(self, persist_directory: Path, embedding_function: object) -> None:
        self.persist_directory = persist_directory
        self.embedding_function = embedding_function
        self._store = Chroma(
            collection_name="personal_knowledge",
            persist_directory=str(persist_directory),
            embedding_function=embedding_function,
        )

    def add_documents(self, documents: list[Document]) -> None:
        ids = [chunk_id_for(doc) for doc in documents]
        self._store.add_documents(documents=documents, ids=ids)

    def similarity_search(self, query: str, k: int) -> list[tuple[Document, float]]:
        return self._store.similarity_search_with_relevance_scores(query, k=k)

    def reset(self) -> None:
        self._store.reset_collection()
