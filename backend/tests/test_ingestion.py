from pathlib import Path

from langchain_core.documents import Document
from pypdf import PdfWriter

from app.rag import ingestion
from app.rag.ingestion import IngestionService


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
    vector_store = FakeVectorStore()

    result = IngestionService(docs_dir=tmp_path, vector_store=vector_store).run()

    names = {document.metadata["file"] for document in vector_store.documents}
    assert "note.txt" in names
    assert "doc.md" in names
    assert "skip.json" not in names
    assert result.documents_indexed == 2


def test_loads_pdf(tmp_path: Path) -> None:
    pdf_path = tmp_path / "file.pdf"
    writer = PdfWriter()
    writer.add_blank_page(width=72, height=72)
    with pdf_path.open("wb") as handle:
        writer.write(handle)
    vector_store = FakeVectorStore()

    result = IngestionService(docs_dir=tmp_path, vector_store=vector_store).run()

    assert result.documents_indexed == 1
    assert vector_store.reset_called is True


def test_stores_chunk_metadata(tmp_path: Path) -> None:
    (tmp_path / "note.txt").write_text("content " * 300)
    vector_store = FakeVectorStore()
    service = IngestionService(docs_dir=tmp_path, vector_store=vector_store)
    result = service.run()
    assert result.documents_indexed == 1
    assert result.chunks_indexed >= 1
    assert vector_store.reset_called is True
    assert "chunk_id" in vector_store.documents[0].metadata
    assert vector_store.documents[0].metadata["file"] == "note.txt"


def test_resets_index_without_adding_when_no_chunks(tmp_path: Path) -> None:
    (tmp_path / "blank.txt").write_text("   ")
    vector_store = FakeVectorStore()

    result = IngestionService(docs_dir=tmp_path, vector_store=vector_store).run()

    assert result.documents_indexed == 1
    assert result.chunks_indexed == 0
    assert vector_store.reset_called is True
    assert vector_store.documents == []


def test_cli_ingestion_uses_runtime(monkeypatch, capsys) -> None:
    class FakeIngestionService:
        def run(self):
            return ingestion.IngestResponse(documents_indexed=2, chunks_indexed=3)

    class FakeRuntime:
        def ingestion_service(self) -> FakeIngestionService:
            return FakeIngestionService()

    monkeypatch.setattr("app.runtime.get_runtime", lambda: FakeRuntime())

    ingestion.main()

    assert capsys.readouterr().out == "Ingestion complete: 2 documents, 3 chunks.\n"
