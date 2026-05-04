from pathlib import Path

from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_core.documents import Document

SUPPORTED_SUFFIXES = {".txt", ".md", ".pdf"}


def load_document(path: Path) -> list[Document]:
    suffix = path.suffix.lower()
    if suffix == ".txt":
        loader = TextLoader(str(path), autodetect_encoding=True)
    elif suffix == ".md":
        loader = TextLoader(str(path), autodetect_encoding=True)
    elif suffix == ".pdf":
        loader = PyPDFLoader(str(path))
    else:
        return []
    return loader.load()


def load_documents(docs_dir: Path) -> list[Document]:
    loaded: list[Document] = []
    for path in sorted(docs_dir.rglob("*")):
        if not path.is_file() or path.suffix.lower() not in SUPPORTED_SUFFIXES:
            continue
        for document in load_document(path):
            metadata = dict(document.metadata)
            metadata["file"] = path.name
            metadata["path"] = str(path)
            metadata["page"] = metadata.get("page")
            loaded.append(Document(page_content=document.page_content, metadata=metadata))
    return loaded
