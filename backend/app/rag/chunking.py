from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter


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
        metadata = dict(chunk.metadata)
        file_stem = metadata.get("file", "document").rsplit(".", 1)[0]
        metadata["chunk_id"] = f"{file_stem}_{index:03d}"
        cleaned.append(Document(page_content=text, metadata=metadata))
    return cleaned
