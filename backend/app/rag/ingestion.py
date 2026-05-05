from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

from langchain_core.documents import Document

from app.rag.chunking import split_documents
from app.rag.loaders import SUPPORTED_SUFFIXES, load_document
from app.schemas import IngestResponse


class VectorIndex(Protocol):
    def reset(self) -> None:
        ...

    def add_documents(self, documents: list[Document]) -> None:
        ...


@dataclass(slots=True)
class IngestionService:
    docs_dir: Path
    vector_store: VectorIndex
    chunk_size: int = 800
    chunk_overlap: int = 120

    def run(self) -> IngestResponse:
        documents = self._load_documents()
        chunks = self._split_documents(documents)
        self._replace_index(chunks)
        return IngestResponse(
            documents_indexed=len(documents),
            chunks_indexed=len(chunks),
        )

    def _load_documents(self) -> list[Document]:
        loaded: list[Document] = []
        for path in sorted(self.docs_dir.rglob("*")):
            if not path.is_file() or path.suffix.lower() not in SUPPORTED_SUFFIXES:
                continue
            for document in load_document(path):
                loaded.append(self._with_source_metadata(document, path))
        return loaded

    def _with_source_metadata(self, document: Document, path: Path) -> Document:
        metadata = dict(document.metadata)
        metadata["file"] = path.name
        metadata["path"] = str(path)
        metadata["page"] = metadata.get("page")
        return Document(page_content=document.page_content, metadata=metadata)

    def _split_documents(self, documents: list[Document]) -> list[Document]:
        return split_documents(
            documents,
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
        )

    def _replace_index(self, chunks: list[Document]) -> None:
        self.vector_store.reset()
        if chunks:
            self.vector_store.add_documents(chunks)


def main() -> None:
    from app.runtime import get_runtime

    result = get_runtime().ingestion_service().run()
    print(
        f"Ingestion complete: {result.documents_indexed} documents, "
        f"{result.chunks_indexed} chunks."
    )


if __name__ == "__main__":
    main()
