from langchain_core.documents import Document

from app.rag.chunking import split_documents


def test_long_text_chunks_correctly() -> None:
    text = "a" * 1200
    chunks = split_documents([Document(page_content=text, metadata={"file": "resume.md"})])
    assert len(chunks) > 1


def test_short_text_remains_one_chunk() -> None:
    chunks = split_documents([Document(page_content="short text", metadata={"file": "note.txt"})])
    assert len(chunks) == 1


def test_metadata_preserved() -> None:
    chunks = split_documents(
        [Document(page_content="content", metadata={"file": "resume.md", "page": 1})]
    )
    assert chunks[0].metadata["file"] == "resume.md"
    assert chunks[0].metadata["page"] == 1
    assert "chunk_id" in chunks[0].metadata


def test_no_empty_chunks() -> None:
    chunks = split_documents([Document(page_content="   ", metadata={"file": "blank.txt"})])
    assert chunks == []


def test_overlap_exists() -> None:
    text = "".join(str(index % 10) for index in range(1000))
    chunks = split_documents(
        [Document(page_content=text, metadata={"file": "overlap.txt"})],
        chunk_size=200,
        chunk_overlap=50,
    )
    assert len(chunks) > 1
    assert chunks[0].page_content[-50:] in chunks[1].page_content

