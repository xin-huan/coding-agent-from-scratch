"""DeepSeek model adapter."""

from __future__ import annotations

import json
from typing import Any

from coding_agent.agent import ModelReply, TokenUsage, ToolCall
from coding_agent.config import Settings


class ModelError(RuntimeError):
    """Raised when a model response cannot be used by the agent."""

    def __init__(self, message: str, *, usage: TokenUsage | None = None) -> None:
        super().__init__(message)
        self.usage = usage


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
        usage = _response_usage(response)

        tool_calls: list[ToolCall] = []
        for call in message.tool_calls or []:
            arguments = _parse_tool_arguments(
                call.function.arguments,
                call.function.name,
                usage,
            )
            if not isinstance(arguments, dict):
                raise ModelError(
                    f"Tool arguments must be an object: {call.function.name}",
                    usage=usage,
                )
            tool_calls.append(
                ToolCall(
                    id=call.id,
                    name=call.function.name,
                    arguments=arguments,
                )
            )

        return ModelReply(
            content=message.content,
            tool_calls=tuple(tool_calls),
            usage=usage,
        )


def _parse_tool_arguments(
    raw_arguments: object,
    tool_name: str,
    usage: TokenUsage | None,
) -> object:
    if not isinstance(raw_arguments, str):
        raise ModelError(f"Invalid arguments for tool {tool_name}", usage=usage)
    try:
        return json.loads(raw_arguments)
    except json.JSONDecodeError as strict_error:
        stripped = raw_arguments.strip()
        try:
            parsed, _end = json.JSONDecoder().raw_decode(stripped)
        except json.JSONDecodeError as raw_error:
            raise ModelError(
                f"Invalid arguments for tool {tool_name}",
                usage=usage,
            ) from raw_error
        if not isinstance(parsed, dict):
            raise ModelError(
                f"Invalid arguments for tool {tool_name}",
                usage=usage,
            ) from strict_error
        return parsed


def _response_usage(response: Any) -> TokenUsage | None:
    response_usage = getattr(response, "usage", None)
    if response_usage is None:
        return None
    return TokenUsage(
        prompt_tokens=int(getattr(response_usage, "prompt_tokens", 0) or 0),
        completion_tokens=int(getattr(response_usage, "completion_tokens", 0) or 0),
        total_tokens=int(getattr(response_usage, "total_tokens", 0) or 0),
        cache_hit_tokens=int(
            getattr(response_usage, "prompt_cache_hit_tokens", 0) or 0
        ),
    )
