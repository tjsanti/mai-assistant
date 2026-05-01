from app.llms.base import LLMProvider
from app.rag.retriever import RetrievedChunk, Retriever
from app.schemas import ChatResponse, SourceMetadata

UNSUPPORTED_ANSWER = "I don't have enough information in the provided documents to answer that."


class RagAnswerer:
    def __init__(self, retriever: Retriever, provider: LLMProvider) -> None:
        self.retriever = retriever
        self.provider = provider

    def run(self, query: str) -> ChatResponse:
        retrieved = self.retriever.retrieve(query)
        if not retrieved:
            return ChatResponse(
                answer=UNSUPPORTED_ANSWER,
                tool_used="rag_answer",
                sources=[],
                llm_provider=self.provider.provider_name,  # type: ignore[arg-type]
            )

        context = self._build_context(retrieved)
        answer = self.provider.generate(
            [
                {
                    "role": "system",
                    "content": (
                        "Answer only from the retrieved context. Cite source files. "
                        "Do not use outside knowledge. If the context is insufficient, "
                        f"reply exactly with: {UNSUPPORTED_ANSWER}"
                    ),
                },
                {
                    "role": "user",
                    "content": f"Question: {query}\n\nRetrieved context:\n{context}",
                },
            ]
        ).strip()
        return ChatResponse(
            answer=answer or UNSUPPORTED_ANSWER,
            tool_used="rag_answer",
            sources=self._sources_from_chunks(retrieved),
            llm_provider=self.provider.provider_name,  # type: ignore[arg-type]
        )

    def _build_context(self, chunks: list[RetrievedChunk]) -> str:
        lines: list[str] = []
        for chunk in chunks:
            metadata = chunk.document.metadata
            lines.append(
                f"Source: {metadata.get('file')} | Page: {metadata.get('page')} | "
                f"Chunk: {metadata.get('chunk_id')}\n{chunk.document.page_content}"
            )
        return "\n\n".join(lines)

    def _sources_from_chunks(self, chunks: list[RetrievedChunk]) -> list[SourceMetadata]:
        return [
            SourceMetadata(
                file=chunk.document.metadata.get("file", "unknown"),
                page=chunk.document.metadata.get("page"),
                chunk_id=chunk.document.metadata.get("chunk_id", "unknown"),
                score=chunk.score,
            )
            for chunk in chunks
        ]
