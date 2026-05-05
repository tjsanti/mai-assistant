from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from app.rag.chunks import with_chunk_id


def split_documents(
    documents: list[Document],
    chunk_size: int = 800,
    chunk_overlap: int = 120,
) -> list[Document]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )
    chunks = splitter.split_documents(documents)
    cleaned: list[Document] = []
    for index, chunk in enumerate(chunks):
        text = chunk.page_content.strip()
        if not text:
            continue
        cleaned.append(
            with_chunk_id(Document(page_content=text, metadata=chunk.metadata), index)
        )
    return cleaned
