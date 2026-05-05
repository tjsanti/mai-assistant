from dataclasses import dataclass
from typing import Any

from langchain_core.documents import Document

from app.schemas import SourceMetadata


def build_chunk_id(document: Document, index: int) -> str:
    metadata = ChunkMetadata.from_document(document)
    file_stem = metadata.file.rsplit(".", 1)[0]
    return f"{file_stem}_{index:03d}"


def with_chunk_id(document: Document, index: int) -> Document:
    metadata = dict(document.metadata)
    metadata["chunk_id"] = build_chunk_id(document, index)
    return Document(page_content=document.page_content, metadata=metadata)


def chunk_id_for(document: Document) -> str:
    return ChunkMetadata.from_document(document).chunk_id


@dataclass(frozen=True, slots=True)
class ChunkMetadata:
    file: str
    path: str | None
    page: int | None
    chunk_id: str

    @classmethod
    def from_document(cls, document: Document) -> "ChunkMetadata":
        metadata = document.metadata
        return cls(
            file=str(metadata.get("file", "unknown")),
            path=_optional_str(metadata.get("path")),
            page=_optional_int(metadata.get("page")),
            chunk_id=str(metadata.get("chunk_id", "unknown")),
        )

    def to_source(self, score: float | None = None) -> SourceMetadata:
        return SourceMetadata(
            file=self.file,
            page=self.page,
            chunk_id=self.chunk_id,
            score=score,
        )


@dataclass(frozen=True, slots=True)
class RetrievedDocumentChunk:
    content: str
    metadata: ChunkMetadata
    score: float

    @classmethod
    def from_document(cls, document: Document, score: float) -> "RetrievedDocumentChunk":
        return cls(
            content=document.page_content,
            metadata=ChunkMetadata.from_document(document),
            score=score,
        )

    def format_for_context(self) -> str:
        return (
            f"Source: {self.metadata.file} | Page: {self.metadata.page} | "
            f"Chunk: {self.metadata.chunk_id}\n{self.content}"
        )

    def to_source(self) -> SourceMetadata:
        return self.metadata.to_source(score=self.score)


def _optional_str(value: Any) -> str | None:
    if value is None:
        return None
    return str(value)


def _optional_int(value: Any) -> int | None:
    if value is None:
        return None
    return int(value)
