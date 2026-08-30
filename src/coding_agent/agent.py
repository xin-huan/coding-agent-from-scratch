"""The core model-tool-model loop."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Callable, Protocol

from coding_agent.checkpoint import CheckpointError, CheckpointStore
from coding_agent.tools.registry import ToolRegistry
from coding_agent.trace import NullTrace, Trace
from coding_agent.workspace import Workspace


SYSTEM_PROMPT = """You are a coding agent working in one local project workspace.
Inspect relevant files before editing. Prefer apply_patch for existing files.
Run an appropriate test or command after making changes.
Never access secrets or paths outside the workspace.
Use the runtime task state to avoid repeating completed work.
If a missing user decision would make edits unsafe or lead to materially
different implementations, call ask_user before modifying files. Do not ask
about details that can be safely inferred from the project and task.
Use only the provided tools and report results honestly in concise Chinese.
"""

FINAL_ANSWER_PROTOCOL_MARKERS = (
    "<tool_call",
    "</tool_call",
    "<function=",
)
DEFAULT_MAX_STEPS = 16
ASK_USER_DEFINITION = {
    "type": "function",
    "function": {
        "name": "ask_user",
        "description": (
            "Pause and ask one concise clarification question when a required "
            "user decision is missing. Use this before modifying files."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "question": {
                    "type": "string",
                    "description": "One concrete question needed to continue safely.",
                }
            },
            "required": ["question"],
            "additionalProperties": False,
        },
    },
}
PROCEED_TASK_DEFINITION = {
    "type": "function",
    "function": {
        "name": "proceed_task",
        "description": "Confirm that the task is concrete enough to start safely.",
        "parameters": {
            "type": "object",
            "properties": {},
            "additionalProperties": False,
        },
    },
}
CLARIFICATION_GATE_PROMPT = """Decide whether this coding request is actionable.
Call ask_user if the target, desired outcome, or an essential user choice is
missing and different choices would lead to materially different edits.
Call proceed_task only when the request is concrete enough to begin safely.
For example, 'optimize this project' requires clarification, while 'reduce list
loading time without changing the public API' can proceed. Call exactly one of
the two control actions and do not answer with plain text.
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

    def to_data(self) -> dict[str, object]:
        return {
            "goal": self.goal,
            "phase": self.phase,
            "modified_files": self.modified_files,
            "latest_command": self.latest_command,
            "last_error": self.last_error,
            "changes_pending_verification": self.changes_pending_verification,
            "full_tests_passed": self.full_tests_passed,
        }

    @classmethod
    def from_data(cls, data: object) -> "TaskState":
        if not isinstance(data, dict):
            raise CheckpointError("Invalid checkpoint task state")
        modified_files = data.get("modified_files")
        if not isinstance(modified_files, list) or not all(
            isinstance(path, str) for path in modified_files
        ):
            raise CheckpointError("Invalid checkpoint modified files")
        return cls(
            goal=str(data.get("goal", "")),
            phase=str(data.get("phase", "inspect")),
            modified_files=modified_files,
            latest_command=str(data.get("latest_command", "not run")),
            last_error=str(data.get("last_error", "none")),
            changes_pending_verification=bool(
                data.get("changes_pending_verification", False)
            ),
            full_tests_passed=bool(data.get("full_tests_passed", False)),
        )


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


def _needs_clarification_gate(task: str) -> bool:
    normalized = " ".join(task.lower().split())
    vague_markers = (
        "优化这个项目",
        "优化项目",
        "改进这个项目",
        "完善这个项目",
        "帮我优化",
        "optimize this project",
        "improve this project",
        "make this project better",
    )
    specific_markers = (
        "性能",
        "速度",
        "内存",
        "可读性",
        "安全",
        "bug",
        "错误",
        "异常",
        "测试",
        "功能",
        "接口",
        "文件",
        ".py",
        ".js",
        ".ts",
    )
    return (
        len(normalized) <= 100
        and any(marker in normalized for marker in vague_markers)
        and not any(marker in normalized for marker in specific_markers)
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
        checkpoint_store: CheckpointStore | None = None,
    ) -> None:
        self.model = model
        self.workspace = workspace
        self.tools = ToolRegistry(workspace)
        self.max_steps = max_steps
        self.on_event = on_event or (lambda _message: None)
        self.trace = trace or NullTrace()
        self.checkpoint_store = checkpoint_store
        self._pending_task: str | None = None

    @property
    def awaiting_clarification(self) -> bool:
        return self._pending_task is not None

    def run(self, task: str) -> str:
        if self._pending_task is not None:
            task = (
                f"Original task:\n{self._pending_task}\n\n"
                f"User clarification:\n{task}"
            )
            self._pending_task = None
        self.trace.record("task_start", task=task)
        messages: list[dict[str, object]] = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": task},
        ]
        state = TaskState(task)

        if _needs_clarification_gate(task):
            question = self._run_clarification_gate(task)
            if question is not None:
                return question

        return self._continue(task, messages, state, step=1)

    def _run_clarification_gate(self, task: str) -> str | None:
        messages: list[dict[str, object]] = [
            {"role": "system", "content": CLARIFICATION_GATE_PROMPT},
            {"role": "user", "content": task},
        ]
        self.trace.record("model_request", step=0, clarification_gate=True)
        try:
            reply = self.model.complete(
                messages,
                [ASK_USER_DEFINITION, PROCEED_TASK_DEFINITION],
            )
        except Exception as error:
            self.trace.record(
                "task_error",
                step=0,
                error_type=type(error).__name__,
            )
            raise AgentError(f"Clarification check failed: {error}") from error
        self.trace.record(
            "model_reply",
            step=0,
            content=reply.content,
            tools=[call.name for call in reply.tool_calls],
        )

        clarification = next(
            (call for call in reply.tool_calls if call.name == "ask_user"),
            None,
        )
        if clarification is not None:
            question = str(clarification.arguments.get("question", "")).strip()
            if not question:
                raise AgentError("Model requested clarification without a question")
            return self._pause_for_clarification(task, question, step=0)
        if len(reply.tool_calls) == 1 and reply.tool_calls[0].name == "proceed_task":
            self.trace.record("clarification_passed", step=0)
            return None
        raise AgentError("Model did not make a valid clarification decision")

    def resume(self) -> str:
        if self.checkpoint_store is None:
            raise AgentError("Checkpoint recovery is not configured")
        try:
            data = self.checkpoint_store.load()
            if data.get("version") != 1:
                raise CheckpointError("Unsupported checkpoint version")
            if data.get("workspace") != str(self.workspace.root):
                raise CheckpointError("Checkpoint belongs to a different workspace")
            task = data.get("task")
            if not isinstance(task, str):
                raise CheckpointError("Invalid checkpoint task")
            status = data.get("status", "running")
            if status == "clarification":
                question = data.get("question")
                if not isinstance(question, str) or not question:
                    raise CheckpointError("Invalid checkpoint clarification")
                self._pending_task = task
                self.trace.record(
                    "task_resumed",
                    task=task,
                    reason="clarification_needed",
                )
                return question
            if status != "running":
                raise CheckpointError("Invalid checkpoint status")
            messages_data = data.get("messages")
            step = data.get("step")
            pending_data = data.get("pending_calls", [])
            next_call_index = data.get("next_call_index", 0)
            if not isinstance(messages_data, list):
                raise CheckpointError("Invalid checkpoint messages")
            if not isinstance(step, int) or not isinstance(next_call_index, int):
                raise CheckpointError("Invalid checkpoint position")
            if not all(isinstance(message, dict) for message in messages_data):
                raise CheckpointError("Invalid checkpoint messages")
            if not isinstance(pending_data, list):
                raise CheckpointError("Invalid checkpoint pending calls")
            messages = [dict(message) for message in messages_data]
            state = TaskState.from_data(data.get("state"))
            pending_calls = tuple(self._tool_call_from_data(item) for item in pending_data)
        except CheckpointError as error:
            raise AgentError(str(error)) from error

        self.trace.record("task_resumed", task=task, step=step)
        return self._continue(
            task,
            messages,
            state,
            step=step,
            pending_calls=pending_calls,
            next_call_index=next_call_index,
        )

    def _continue(
        self,
        task: str,
        messages: list[dict[str, object]],
        state: TaskState,
        *,
        step: int,
        pending_calls: tuple[ToolCall, ...] = (),
        next_call_index: int = 0,
    ) -> str:
        if pending_calls:
            self._execute_tool_calls(
                task,
                messages,
                state,
                step,
                pending_calls,
                start_index=next_call_index,
            )
            if state.ready_to_finalize:
                return self._finalize_after_verification(messages, state, step + 1)
            step += 1

        while step <= self.max_steps:
            self.trace.record("model_request", step=step)
            request_messages = [
                *messages,
                state.message(self.max_steps - step + 1),
            ]
            try:
                reply = self.model.complete(
                    request_messages,
                    [*self.tools.definitions, ASK_USER_DEFINITION],
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
            clarification = next(
                (call for call in reply.tool_calls if call.name == "ask_user"),
                None,
            )
            if clarification is not None:
                question = str(clarification.arguments.get("question", "")).strip()
                if not question:
                    error = AgentError("Model requested clarification without a question")
                    self.trace.record("task_error", step=step, error=str(error))
                    raise error
                if state.modified_files:
                    error = AgentError("Model requested clarification after modifying files")
                    self.trace.record("task_error", step=step, error=str(error))
                    raise error
                return self._pause_for_clarification(task, question, step=step)
            if not reply.tool_calls:
                if reply.content:
                    if self._contains_tool_protocol(reply.content):
                        return self._retry_final_answer(
                            request_messages,
                            reply.content,
                            step,
                        )
                    self._clear_checkpoint()
                    self.trace.record("task_complete", step=step, answer=reply.content)
                    return reply.content
                error = AgentError("Model returned neither a tool call nor an answer")
                self.trace.record("task_error", step=step, error=str(error))
                raise error

            messages.append(self._assistant_message(reply))
            self._execute_tool_calls(
                task,
                messages,
                state,
                step,
                reply.tool_calls,
            )

            if state.ready_to_finalize:
                return self._finalize_after_verification(messages, state, step + 1)
            step += 1

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

    def _execute_tool_calls(
        self,
        task: str,
        messages: list[dict[str, object]],
        state: TaskState,
        step: int,
        calls: tuple[ToolCall, ...],
        *,
        start_index: int = 0,
    ) -> None:
        for index in range(start_index, len(calls)):
            call = calls[index]
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
            self._save_checkpoint(
                task,
                messages,
                state,
                step,
                calls,
                next_call_index=index + 1,
            )

    def _finalize_after_verification(
        self,
        messages: list[dict[str, object]],
        state: TaskState,
        step: int,
    ) -> str:
        return self._finalize(
            messages,
            state,
            step,
            (
                "Full test suite passed after the latest file changes. "
                "Do not call more tools. Give a concise, honest final "
                "report in Chinese based on the completed work and tests."
            ),
            reason="verification_passed",
        )

    def _save_checkpoint(
        self,
        task: str,
        messages: list[dict[str, object]],
        state: TaskState,
        step: int,
        pending_calls: tuple[ToolCall, ...],
        *,
        next_call_index: int,
    ) -> None:
        if self.checkpoint_store is None:
            return
        self.checkpoint_store.save(
            {
                "version": 1,
                "status": "running",
                "workspace": str(self.workspace.root),
                "task": task,
                "messages": messages,
                "state": state.to_data(),
                "step": step,
                "pending_calls": [self._tool_call_to_data(call) for call in pending_calls],
                "next_call_index": next_call_index,
            }
        )

    def _save_clarification_checkpoint(self, task: str, question: str) -> None:
        if self.checkpoint_store is None:
            return
        self.checkpoint_store.save(
            {
                "version": 1,
                "status": "clarification",
                "workspace": str(self.workspace.root),
                "task": task,
                "question": question,
            }
        )

    def _pause_for_clarification(
        self,
        task: str,
        question: str,
        *,
        step: int,
    ) -> str:
        self._pending_task = task
        self._save_clarification_checkpoint(task, question)
        self.trace.record(
            "task_paused",
            step=step,
            reason="clarification_needed",
            question=question,
        )
        return question

    def _clear_checkpoint(self) -> None:
        if self.checkpoint_store is not None:
            self.checkpoint_store.clear()

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

        self._clear_checkpoint()
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
                self._clear_checkpoint()
                self.trace.record(
                    "task_complete",
                    step=step,
                    answer=fallback_answer,
                )
                return fallback_answer
            error = AgentError("Model did not provide a clean final answer")
            self.trace.record("task_error", step=step, error=str(error))
            raise error

        self._clear_checkpoint()
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
    def _tool_call_to_data(call: ToolCall) -> dict[str, object]:
        return {
            "id": call.id,
            "name": call.name,
            "arguments": call.arguments,
        }

    @staticmethod
    def _tool_call_from_data(data: object) -> ToolCall:
        if not isinstance(data, dict):
            raise CheckpointError("Invalid checkpoint tool call")
        call_id = data.get("id")
        name = data.get("name")
        arguments = data.get("arguments")
        if not isinstance(call_id, str) or not isinstance(name, str):
            raise CheckpointError("Invalid checkpoint tool identity")
        if not isinstance(arguments, dict):
            raise CheckpointError("Invalid checkpoint tool arguments")
        return ToolCall(call_id, name, dict(arguments))

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
