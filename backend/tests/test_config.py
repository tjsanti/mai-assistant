from pathlib import Path

import pytest

from app.config import Settings


def test_loads_defaults() -> None:
    settings = Settings()
    assert settings.openai_model == "gpt-4.1-mini"
    assert settings.general_llm_provider == "openai"
    assert settings.rag_llm_provider == "ollama"
    assert settings.docs_dir == Path("../docs")


def test_reads_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OPENAI_MODEL", "gpt-4o-mini")
    monkeypatch.setenv("RETRIEVAL_TOP_K", "7")
    settings = Settings()
    assert settings.openai_model == "gpt-4o-mini"
    assert settings.retrieval_top_k == 7


def test_supports_separate_providers(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("GENERAL_LLM_PROVIDER", "openai")
    monkeypatch.setenv("RAG_LLM_PROVIDER", "ollama")
    settings = Settings()
    assert settings.general_llm_provider == "openai"
    assert settings.rag_llm_provider == "ollama"


def test_invalid_provider_raises_clear_error(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("GENERAL_LLM_PROVIDER", "bad")
    with pytest.raises(ValueError, match="Unsupported GENERAL_LLM_PROVIDER"):
        Settings()


def test_openai_rag_requires_explicit_opt_in(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("RAG_LLM_PROVIDER", "openai")
    monkeypatch.delenv("ALLOW_CLOUD_RAG_CONTEXT", raising=False)
    with pytest.raises(ValueError, match="ALLOW_CLOUD_RAG_CONTEXT=true"):
        Settings()

