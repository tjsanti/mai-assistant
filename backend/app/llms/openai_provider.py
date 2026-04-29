from openai import OpenAI

from app.llms.base import LLMProvider, ToolSelectionProvider
from app.schemas import ToolName


class OpenAIProvider(LLMProvider, ToolSelectionProvider):
    provider_name = "openai"

    def __init__(self, api_key: str, model: str) -> None:
        self.model = model
        self.client = OpenAI(api_key=api_key)

    def generate(self, messages: list[dict], **kwargs: object) -> str:
        response = self.client.responses.create(
            model=self.model,
            input=messages,
            **kwargs,
        )
        return response.output_text

    def choose_tool(self, message: str, tools: list[ToolName]) -> ToolName:
        response = self.client.responses.create(
            model=self.model,
            input=[
                {
                    "role": "system",
                    "content": (
                        "Choose the best tool for the user's request. "
                        "Use rag_answer for personal documents, notes, resume, "
                        "projects, or anything grounded in local docs. "
                        "Use general_response for everything else."
                    ),
                },
                {"role": "user", "content": message},
            ],
            tools=[
                {
                    "type": "function",
                    "name": tool_name,
                    "description": f"Select the {tool_name} tool.",
                    "parameters": {"type": "object", "properties": {}, "required": []},
                }
                for tool_name in tools
            ],
            tool_choice="required",
        )
        for item in response.output:
            if getattr(item, "type", None) == "function_call":
                tool_name = item.name
                if tool_name in tools:
                    return tool_name
        raise ValueError("OpenAI did not return a valid tool choice.")

