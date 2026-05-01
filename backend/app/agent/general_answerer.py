from app.llms.base import LLMProvider
from app.schemas import ChatResponse


class GeneralAnswerer:
    def __init__(self, provider: LLMProvider) -> None:
        self.provider = provider

    def run(self, query: str) -> ChatResponse:
        answer = self.provider.generate([{"role": "user", "content": query}])
        return ChatResponse(
            answer=answer,
            tool_used="general_response",
            llm_provider=self.provider.provider_name,  # type: ignore[arg-type]
            sources=[],
        )
