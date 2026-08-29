"""The core model-tool-model loop."""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Callable, Protocol

from coding_agent.tools.registry import ToolRegistry
from coding_agent.trace import NullTrace, Trace
from coding_agent.workspace import Workspace


SYSTEM_PROMPT = """You are a coding agent working in one local project workspace.
Inspect relevant files before editing. Prefer apply_patch for existing files.
Run an appropriate test or command after making changes.
Never access secrets or paths outside the workspace.
Use only the provided tools and report results honestly in concise Chinese.
"""


@dataclass(frozen=True)
class ToolCall:
    id: str
    name: str
    arguments: dict[str, object]


@dataclass(frozen=True)
class ModelReply:
    content: str | None
    tool_calls: tuple[ToolCall, ...] = ()


class Model(Protocol):
    def complete(
        self, messages: list[dict[str, object]], tools: list[dict[str, object]]
    ) -> ModelReply: ...


class AgentError(RuntimeError):
    """Raised when the agent cannot finish safely."""


class Agent:
    def __init__(
        self,
        model: Model,
        workspace: Workspace,
        *,
        max_steps: int = 8,
        on_event: Callable[[str], None] | None = None,
        trace: Trace | None = None,
    ) -> None:
        self.model = model
        self.tools = ToolRegistry(workspace)
        self.max_steps = max_steps
        self.on_event = on_event or (lambda _message: None)
        self.trace = trace or NullTrace()

    def run(self, task: str) -> str:
        self.trace.record("task_start", task=task)
        messages: list[dict[str, object]] = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": task},
        ]

        for step in range(1, self.max_steps + 1):
            self.trace.record("model_request", step=step)
            try:
                reply = self.model.complete(messages, self.tools.definitions)
            except Exception as error:
                self.trace.record(
                    "task_error",
                    step=step,
                    error_type=type(error).__name__,
                )
                raise AgentError(f"Model request failed: {error}") from error

            self.trace.record(
                "model_reply",
                step=step,
                content=reply.content,
                tools=[call.name for call in reply.tool_calls],
            )
            if not reply.tool_calls:
                if reply.content:
                    self.trace.record("task_complete", step=step, answer=reply.content)
                    return reply.content
                error = AgentError("Model returned neither a tool call nor an answer")
                self.trace.record("task_error", step=step, error=str(error))
                raise error

            messages.append(self._assistant_message(reply))
            for call in reply.tool_calls:
                self.on_event(f"[工具] {call.name}")
                self.trace.record(
                    "tool_start",
                    step=step,
                    tool=call.name,
                    arguments=call.arguments,
                )
                try:
                    result = self.tools.execute(call.name, call.arguments)
                except (OSError, ValueError) as error:
                    result = f"ERROR: {error}"
                    success = False
                else:
                    success = True
                self.trace.record(
                    "tool_result",
                    step=step,
                    tool=call.name,
                    success=success,
                    result=result,
                )
                messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": call.id,
                        "content": result,
                    }
                )

        error = AgentError(f"Agent exceeded the maximum of {self.max_steps} steps")
        self.trace.record("task_error", step=self.max_steps, error=str(error))
        raise error

    @staticmethod
    def _assistant_message(reply: ModelReply) -> dict[str, object]:
        return {
            "role": "assistant",
            "content": reply.content,
            "tool_calls": [
                {
                    "id": call.id,
                    "type": "function",
                    "function": {
                        "name": call.name,
                        "arguments": json.dumps(call.arguments, ensure_ascii=False),
                    },
                }
                for call in reply.tool_calls
            ],
        }
