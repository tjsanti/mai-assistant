from pathlib import Path

from langchain_core.documents import Document
from pypdf import PdfWriter

from app.rag.ingestion import IngestionService
from app.rag.loaders import load_documents


class FakeVectorStore:
    def __init__(self) -> None:
        self.reset_called = False
        self.documents: list[Document] = []

    def reset(self) -> None:
        self.reset_called = True

    def add_documents(self, documents: list[Document]) -> None:
        self.documents.extend(documents)


def test_loads_txt_md_and_ignores_unsupported(tmp_path: Path) -> None:
    (tmp_path / "note.txt").write_text("text file")
    (tmp_path / "doc.md").write_text("# heading")
    (tmp_path / "skip.json").write_text("{}")
    documents = load_documents(tmp_path)
    names = {document.metadata["file"] for document in documents}
    assert "note.txt" in names
    assert "doc.md" in names
    assert "skip.json" not in names


def test_loads_pdf(tmp_path: Path) -> None:
    pdf_path = tmp_path / "file.pdf"
    writer = PdfWriter()
    writer.add_blank_page(width=72, height=72)
    with pdf_path.open("wb") as handle:
        writer.write(handle)
    documents = load_documents(tmp_path)
    assert any(document.metadata["file"] == "file.pdf" for document in documents)


def test_stores_chunk_metadata(tmp_path: Path) -> None:
    (tmp_path / "note.txt").write_text("content " * 300)
    vector_store = FakeVectorStore()
    service = IngestionService(docs_dir=tmp_path, vector_store=vector_store)  # type: ignore[arg-type]
    result = service.run()
    assert result.documents_indexed == 1
    assert result.chunks_indexed >= 1
    assert vector_store.reset_called is True
    assert "chunk_id" in vector_store.documents[0].metadata
