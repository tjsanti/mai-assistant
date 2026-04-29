from dataclasses import dataclass
from typing import Protocol

from app.schemas import ToolName


class LLMProvider(Protocol):
    provider_name: str

    def generate(self, messages: list[dict], **kwargs: object) -> str:
        ...


class ToolSelectionProvider(Protocol):
    provider_name: str

    def choose_tool(self, message: str, tools: list[ToolName]) -> ToolName:
        ...


@dataclass(slots=True)
class GeneratedToolCall:
    tool_name: ToolName

