from dataclasses import dataclass

from app.config import Settings
from app.rag.chunking import split_documents
from app.rag.embeddings import get_embeddings
from app.rag.loaders import load_documents
from app.rag.vector_store import VectorStoreClient
from app.schemas import IngestResponse


@dataclass(slots=True)
class IngestionService:
    docs_dir: object
    vector_store: VectorStoreClient

    @classmethod
    def from_settings(cls, settings: Settings) -> "IngestionService":
        embeddings = get_embeddings(settings.embedding_model)
        vector_store = VectorStoreClient(settings.vector_db_dir, embeddings)
        return cls(docs_dir=settings.docs_dir, vector_store=vector_store)

    def run(self) -> IngestResponse:
        documents = load_documents(self.docs_dir)
        chunks = split_documents(documents)
        self.vector_store.reset()
        if chunks:
            self.vector_store.add_documents(chunks)
        return IngestResponse(
            documents_indexed=len(documents),
            chunks_indexed=len(chunks),
        )


def main() -> None:
    from app.config import get_settings

    result = IngestionService.from_settings(get_settings()).run()
    print(
        f"Ingestion complete: {result.documents_indexed} documents, "
        f"{result.chunks_indexed} chunks."
    )


if __name__ == "__main__":
    main()

