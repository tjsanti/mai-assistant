from unittest.mock import Mock, patch

import pytest

from app.config import Settings
from app.llms.ollama_provider import OllamaProvider
from app.llms.openai_provider import OpenAIProvider
from app.llms.router import ProviderRegistry


def test_openai_provider_conforms_to_interface() -> None:
    with patch("app.llms.openai_provider.OpenAI") as mock_client:
        instance = mock_client.return_value
        instance.responses.create.return_value.output_text = "hello"
        provider = OpenAIProvider(api_key="key", model="gpt-4.1-mini")
        assert provider.generate([{"role": "user", "content": "hi"}]) == "hello"


def test_ollama_provider_conforms_to_interface() -> None:
    with patch("app.llms.ollama_provider.httpx.post") as mock_post:
        mock_response = Mock()
        mock_response.json.return_value = {"message": {"content": "hello"}}
        mock_response.raise_for_status.return_value = None
        mock_post.return_value = mock_response
        provider = OllamaProvider(base_url="http://localhost:11434", model="llama3.1:8b")
        assert provider.generate([{"role": "user", "content": "hi"}]) == "hello"


def test_invalid_provider_raises_clear_error() -> None:
    registry = ProviderRegistry(Settings())
    with pytest.raises(ValueError, match="Unsupported provider"):
        registry.get_provider("bad")

