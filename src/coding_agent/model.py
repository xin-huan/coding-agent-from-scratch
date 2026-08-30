"""DeepSeek model adapter."""

from __future__ import annotations

import json
from typing import Any

from coding_agent.agent import ModelReply, ToolCall
from coding_agent.config import Settings


class ModelError(RuntimeError):
    """Raised when a model response cannot be used by the agent."""


class DeepSeekModel:
    def __init__(self, settings: Settings, *, client: Any = None) -> None:
        if client is None:
            from openai import OpenAI

            client = OpenAI(api_key=settings.api_key, base_url=settings.base_url)
        self.client = client
        self.model = settings.model

    def complete(
        self, messages: list[dict[str, object]], tools: list[dict[str, object]]
    ) -> ModelReply:
        request: dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "extra_body": {"thinking": {"type": "disabled"}},
        }
        if tools:
            request["tools"] = tools
        response = self.client.chat.completions.create(**request)
        message = response.choices[0].message

        tool_calls: list[ToolCall] = []
        for call in message.tool_calls or []:
            try:
                arguments = json.loads(call.function.arguments)
            except (TypeError, json.JSONDecodeError) as error:
                raise ModelError(f"Invalid arguments for tool {call.function.name}") from error
            if not isinstance(arguments, dict):
                raise ModelError(f"Tool arguments must be an object: {call.function.name}")
            tool_calls.append(
                ToolCall(
                    id=call.id,
                    name=call.function.name,
                    arguments=arguments,
                )
            )

        return ModelReply(content=message.content, tool_calls=tuple(tool_calls))
