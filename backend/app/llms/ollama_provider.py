import httpx

from app.llms.base import LLMProvider


class OllamaProvider(LLMProvider):
    provider_name = "ollama"

    def __init__(self, base_url: str, model: str, timeout: float = 120.0) -> None:
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.timeout = timeout

    def generate(self, messages: list[dict], **kwargs: object) -> str:
        payload = {
            "model": self.model,
            "messages": messages,
            "stream": False,
        }
        payload.update(kwargs)
        response = httpx.post(
            f"{self.base_url}/api/chat",
            json=payload,
            timeout=self.timeout,
        )
        response.raise_for_status()
        body = response.json()
        return body["message"]["content"]

