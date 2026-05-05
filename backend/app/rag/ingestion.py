from dataclasses import dataclass

from app.rag.chunking import split_documents
from app.rag.loaders import load_documents
from app.rag.vector_store import VectorStoreClient
from app.schemas import IngestResponse


@dataclass(slots=True)
class IngestionService:
    docs_dir: object
    vector_store: VectorStoreClient

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
    from app.runtime import get_runtime

    result = get_runtime().ingestion_service().run()
    print(
        f"Ingestion complete: {result.documents_indexed} documents, "
        f"{result.chunks_indexed} chunks."
    )


if __name__ == "__main__":
    main()
