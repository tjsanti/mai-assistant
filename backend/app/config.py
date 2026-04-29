from functools import lru_cache
from pathlib import Path

from pydantic import Field, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    openai_api_key: str = Field(default="", alias="OPENAI_API_KEY")
    openai_model: str = Field(default="gpt-4.1-mini", alias="OPENAI_MODEL")
    ollama_base_url: str = Field(default="http://localhost:11434", alias="OLLAMA_BASE_URL")
    ollama_model: str = Field(default="llama3.1:8b", alias="OLLAMA_MODEL")
    general_llm_provider: str = Field(default="openai", alias="GENERAL_LLM_PROVIDER")
    rag_llm_provider: str = Field(default="ollama", alias="RAG_LLM_PROVIDER")
    allow_cloud_rag_context: bool = Field(default=False, alias="ALLOW_CLOUD_RAG_CONTEXT")
    embedding_model: str = Field(
        default="BAAI/bge-small-en-v1.5",
        alias="EMBEDDING_MODEL",
    )
    docs_dir: Path = Field(default=Path("../docs"), alias="DOCS_DIR")
    vector_db_dir: Path = Field(default=Path("../.vector_db"), alias="VECTOR_DB_DIR")
    retrieval_top_k: int = Field(default=5, alias="RETRIEVAL_TOP_K")

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        populate_by_name=True,
        extra="ignore",
    )

    @model_validator(mode="after")
    def validate_providers(self) -> "Settings":
        valid = {"openai", "ollama"}
        if self.general_llm_provider not in valid:
            raise ValueError(
                f"Unsupported GENERAL_LLM_PROVIDER '{self.general_llm_provider}'. "
                "Expected one of: openai, ollama."
            )
        if self.rag_llm_provider not in valid:
            raise ValueError(
                f"Unsupported RAG_LLM_PROVIDER '{self.rag_llm_provider}'. "
                "Expected one of: openai, ollama."
            )
        if self.rag_llm_provider == "openai" and not self.allow_cloud_rag_context:
            raise ValueError(
                "RAG_LLM_PROVIDER=openai requires ALLOW_CLOUD_RAG_CONTEXT=true."
            )
        return self


@lru_cache
def get_settings() -> Settings:
    return Settings()

