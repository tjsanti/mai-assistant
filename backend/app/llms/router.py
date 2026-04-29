from app.config import Settings
from app.llms.base import LLMProvider, ToolSelectionProvider
from app.llms.ollama_provider import OllamaProvider
from app.llms.openai_provider import OpenAIProvider


class ProviderRegistry:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings

    def get_provider(self, provider_name: str) -> LLMProvider:
        if provider_name == "openai":
            return OpenAIProvider(
                api_key=self.settings.openai_api_key,
                model=self.settings.openai_model,
            )
        if provider_name == "ollama":
            return OllamaProvider(
                base_url=self.settings.ollama_base_url,
                model=self.settings.ollama_model,
            )
        raise ValueError(f"Unsupported provider '{provider_name}'.")

    def get_general_provider(self) -> LLMProvider:
        return self.get_provider(self.settings.general_llm_provider)

    def get_rag_provider(self) -> LLMProvider:
        return self.get_provider(self.settings.rag_llm_provider)

    def get_tool_selection_provider(self) -> ToolSelectionProvider:
        provider = self.get_general_provider()
        if not hasattr(provider, "choose_tool"):
            raise ValueError(
                f"Provider '{provider.provider_name}' does not support tool selection."
            )
        return provider  # type: ignore[return-value]

