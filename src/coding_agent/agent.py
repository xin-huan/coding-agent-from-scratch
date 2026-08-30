"""The core model-tool-model loop."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Callable, Protocol

from coding_agent.tools.registry import ToolRegistry
from coding_agent.trace import NullTrace, Trace
from coding_agent.workspace import Workspace


SYSTEM_PROMPT = """You are a coding agent working in one local project workspace.
Inspect relevant files before editing. Prefer apply_patch for existing files.
Run an appropriate test or command after making changes.
Never access secrets or paths outside the workspace.
Use the runtime task state to avoid repeating completed work.
Use only the provided tools and report results honestly in concise Chinese.
"""

FINAL_ANSWER_PROTOCOL_MARKERS = (
    "<tool_call",
    "</tool_call",
    "<function=",
)
DEFAULT_MAX_STEPS = 16


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


@dataclass
class TaskState:
    goal: str
    phase: str = "inspect"
    modified_files: list[str] = field(default_factory=list)
    latest_command: str = "not run"
    last_error: str = "none"
    changes_pending_verification: bool = False
    full_tests_passed: bool = False

    @property
    def ready_to_finalize(self) -> bool:
        return self.full_tests_passed and not self.changes_pending_verification

    def update(self, call: ToolCall, result: str, success: bool) -> None:
        if not success:
            self.phase = "finalize" if self.ready_to_finalize else "repair"
            self.last_error = result[:200]
            return

        if call.name in {"write_file", "apply_patch"}:
            path = str(call.arguments.get("path", "unknown"))
            if path not in self.modified_files:
                self.modified_files.append(path)
            self.changes_pending_verification = True
            self.full_tests_passed = False
            self.phase = "verify"
            self.last_error = "none"
        elif call.name == "run_command":
            if "Exit code: 0" in result:
                self.latest_command = "passed (exit 0)"
                if self.changes_pending_verification and _is_full_test_command(call):
                    self.changes_pending_verification = False
                    self.full_tests_passed = True
                if self.ready_to_finalize:
                    self.phase = "finalize"
                elif self.changes_pending_verification:
                    self.phase = "verify"
                else:
                    self.phase = "inspect"
                self.last_error = "none"
            else:
                self.latest_command = "failed"
                self.phase = "finalize" if self.ready_to_finalize else "repair"
                self.last_error = result[:200]

    def message(self, remaining_rounds: int) -> dict[str, object]:
        files = ", ".join(self.modified_files) or "none"
        next_focus = {
            "inspect": "inspect only the files needed for the task",
            "verify": "run focused verification for the changes",
            "repair": "diagnose the latest failure before changing more code",
            "finalize": (
                "Return a final answer now. Do not call another tool unless an "
                "explicit acceptance requirement remains unverified"
            ),
        }[self.phase]
        return {
            "role": "system",
            "content": (
                "<task_state>\n"
                f"Goal: {self.goal}\n"
                f"Phase: {self.phase}\n"
                f"Modified files: {files}\n"
                f"Latest command: {self.latest_command}\n"
                f"Last error: {self.last_error}\n"
                f"Remaining action rounds: {remaining_rounds}\n"
                f"Next focus: {next_focus}\n"
                "</task_state>"
            ),
        }


def _is_full_test_command(call: ToolCall) -> bool:
    if call.name != "run_command":
        return False
    argv = call.arguments.get("argv")
    if not isinstance(argv, list) or not argv:
        return False

    arguments = [str(value) for value in argv]
    program = arguments[0].replace("\\", "/").rsplit("/", 1)[-1].lower()
    if program in {"pytest", "pytest.exe", "py.test", "py.test.exe"}:
        return True
    if not program.startswith("python"):
        return False
    return (
        len(arguments) >= 4
        and arguments[1:3] == ["-m", "unittest"]
        and "discover" in arguments[3:]
    )


class Agent:
    def __init__(
        self,
        model: Model,
        workspace: Workspace,
        *,
        max_steps: int = DEFAULT_MAX_STEPS,
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
        state = TaskState(task)

        for step in range(1, self.max_steps + 1):
            self.trace.record("model_request", step=step)
            request_messages = [
                *messages,
                state.message(self.max_steps - step + 1),
            ]
            try:
                reply = self.model.complete(
                    request_messages,
                    self.tools.definitions,
                )
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
                    if self._contains_tool_protocol(reply.content):
                        return self._retry_final_answer(
                            request_messages,
                            reply.content,
                            step,
                        )
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
                state.update(call, result, success)
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

            if state.ready_to_finalize:
                return self._finalize(
                    messages,
                    state,
                    step + 1,
                    (
                        "Full test suite passed after the latest file changes. "
                        "Do not call more tools. Give a concise, honest final "
                        "report in Chinese based on the completed work and tests."
                    ),
                    reason="verification_passed",
                )

        return self._finalize(
            messages,
            state,
            self.max_steps + 1,
            (
                "The tool budget is exhausted. Do not call more tools. "
                "Give a concise, honest final report based on the results above."
            ),
            reason="tool_budget_exhausted",
        )

    def _finalize(
        self,
        messages: list[dict[str, object]],
        state: TaskState,
        step: int,
        instruction: str,
        *,
        reason: str,
    ) -> str:
        request_messages = [
            *messages,
            state.message(0),
            {"role": "system", "content": instruction},
        ]
        self.trace.record(
            "model_request",
            step=step,
            finalization=True,
            reason=reason,
        )
        try:
            reply = self.model.complete(request_messages, [])
        except Exception as error:
            self.trace.record(
                "task_error",
                step=step,
                error_type=type(error).__name__,
            )
            raise AgentError(f"Final model request failed: {error}") from error

        self.trace.record(
            "model_reply",
            step=step,
            content=reply.content,
            tools=[call.name for call in reply.tool_calls],
        )
        if reply.tool_calls or not reply.content:
            error = AgentError("Model did not provide a final answer")
            self.trace.record("task_error", step=step, error=str(error))
            raise error

        if self._contains_tool_protocol(reply.content):
            return self._retry_final_answer(
                request_messages,
                reply.content,
                step,
                fallback_answer=(
                    self._verified_runtime_summary(state)
                    if reason == "verification_passed"
                    else None
                ),
            )

        self.trace.record("task_complete", step=step, answer=reply.content)
        return reply.content

    @staticmethod
    def _contains_tool_protocol(content: str) -> bool:
        lowered = content.lower()
        if any(marker in lowered for marker in FINAL_ANSWER_PROTOCOL_MARKERS):
            return True
        return "dsml" in lowered and (
            "tool_call" in lowered or "invoke" in lowered
        )

    def _retry_final_answer(
        self,
        messages: list[dict[str, object]],
        rejected_content: str,
        step: int,
        fallback_answer: str | None = None,
    ) -> str:
        self.trace.record(
            "final_answer_rejected",
            step=step,
            reason="tool_protocol_residue",
        )
        retry_messages = [
            *messages,
            {"role": "assistant", "content": rejected_content},
            {
                "role": "system",
                "content": (
                    "Your previous answer contained tool-call protocol markup. "
                    "No tools are available. Return only a concise plain-text "
                    "final report in Chinese. Do not include XML, JSON, DSML, "
                    "tool_calls, or invoke tags."
                ),
            },
        ]
        self.trace.record(
            "model_request",
            step=step,
            finalization=True,
            retry=True,
        )
        try:
            reply = self.model.complete(retry_messages, [])
        except Exception as error:
            self.trace.record(
                "task_error",
                step=step,
                error_type=type(error).__name__,
            )
            raise AgentError(f"Final model retry failed: {error}") from error

        self.trace.record(
            "model_reply",
            step=step,
            content=reply.content,
            tools=[call.name for call in reply.tool_calls],
            retry=True,
        )
        if (
            reply.tool_calls
            or not reply.content
            or self._contains_tool_protocol(reply.content)
        ):
            if fallback_answer is not None:
                self.trace.record(
                    "final_answer_fallback",
                    step=step,
                    reason="verified_model_answer_invalid",
                )
                self.trace.record(
                    "task_complete",
                    step=step,
                    answer=fallback_answer,
                )
                return fallback_answer
            error = AgentError("Model did not provide a clean final answer")
            self.trace.record("task_error", step=step, error=str(error))
            raise error

        self.trace.record("task_complete", step=step, answer=reply.content)
        return reply.content

    @staticmethod
    def _verified_runtime_summary(state: TaskState) -> str:
        files = "、".join(state.modified_files)
        return (
            "任务已完成。\n"
            f"修改文件：{files}。\n"
            "验证结果：修改后的完整测试已通过（退出码 0）。\n"
            "模型最终总结包含无效工具协议，Runtime 已忽略该内容并根据验证记录生成本报告。"
        )

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
