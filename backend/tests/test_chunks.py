from langchain_core.documents import Document

from app.rag.chunks import (
    ChunkMetadata,
    RetrievedDocumentChunk,
    build_chunk_id,
    chunk_id_for,
    with_chunk_id,
)


def test_builds_chunk_id_from_file_stem() -> None:
    document = Document(page_content="content", metadata={"file": "resume.md"})
    assert build_chunk_id(document, 3) == "resume_003"


def test_adds_chunk_id_without_losing_metadata() -> None:
    document = Document(
        page_content="content",
        metadata={"file": "note.txt", "path": "/docs/note.txt", "page": 1},
    )

    chunk = with_chunk_id(document, 4)

    assert chunk_id_for(chunk) == "note_004"
    assert ChunkMetadata.from_document(chunk).path == "/docs/note.txt"
    assert ChunkMetadata.from_document(chunk).page == 1


def test_retrieved_chunk_formats_context_and_source() -> None:
    chunk = RetrievedDocumentChunk(
        content="FAISS experience",
        metadata=ChunkMetadata(
            file="resume.md",
            path="/docs/resume.md",
            page=2,
            chunk_id="resume_001",
        ),
        score=0.82,
    )

    assert chunk.format_for_context() == (
        "Source: resume.md | Page: 2 | Chunk: resume_001\nFAISS experience"
    )
    assert chunk.to_source().file == "resume.md"
    assert chunk.to_source().score == 0.82
