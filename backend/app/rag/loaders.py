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

