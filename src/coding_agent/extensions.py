"""Lifecycle extension hooks for the local coding agent."""

from __future__ import annotations

import json
import hashlib
import re
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Callable, Protocol, Sequence

from coding_agent.project_memory import ProjectMemoryStore
from coding_agent.skills import SkillRegistry
from coding_agent.tools.registry import ToolRegistry
from coding_agent.trace import Trace
from coding_agent.workspace import Workspace


Message = dict[str, object]
ToolDefinition = dict[str, object]
REVIEWER_RESULT_EVENT_PREFIX = "[Reviewer结果] "


@dataclass(frozen=True)
class ExtensionContext:
    task: str
    workspace: Workspace
    trace: Trace
    emit: Callable[[str], None]


@dataclass(frozen=True)
class ToolResult:
    content: str
    success: bool = True


@dataclass(frozen=True)
class ContextOutputRecord:
    output_id: str
    filename: str
    tool: str
    step: int
    characters: int
    lines: int
    success: bool
    summary: str


@dataclass(frozen=True)
class SubagentTask:
    id: str
    role: str
    objective: str
    context: tuple[str, ...]
    constraints: tuple[str, ...]
    allowed_tools: tuple[str, ...]
    expected_output: str
    status: str = "pending"


@dataclass(frozen=True)
class SubagentResult:
    task_id: str
    summary: str
    findings: tuple[dict[str, object], ...]
    artifacts: tuple[str, ...]
    confidence: float
    risks: tuple[str, ...]
    recommended_next_step: str


class AgentExtension(Protocol):
    name: str

    def on_session_start(self, context: ExtensionContext) -> None: ...

    def inject_context(self, context: ExtensionContext) -> list[Message]: ...

    def tool_definitions(self, context: ExtensionContext) -> list[ToolDefinition]: ...

    def before_llm_call(
        self,
        context: ExtensionContext,
        *,
        step: int,
        messages: list[Message],
        tools: list[ToolDefinition],
    ) -> tuple[list[Message], list[ToolDefinition]]: ...

    def after_llm_call(
        self,
        context: ExtensionContext,
        *,
        step: int,
        reply: Any,
    ) -> None: ...

    def on_context_compact(
        self,
        context: ExtensionContext,
        *,
        step: int,
        original_characters: int,
        sent_characters: int,
        compacted_messages: int,
    ) -> None: ...

    def before_tool_call(
        self,
        context: ExtensionContext,
        *,
        step: int,
        call: Any,
    ) -> Any | None: ...

    def execute_tool(
        self,
        context: ExtensionContext,
        *,
        step: int,
        call: Any,
    ) -> ToolResult | None: ...

    def after_tool_call(
        self,
        context: ExtensionContext,
        *,
        step: int,
        call: Any,
        result: ToolResult,
    ) -> ToolResult: ...

    def on_session_end(
        self,
        context: ExtensionContext,
        *,
        answer: str | None,
        error: Exception | None,
        state: Any,
    ) -> None: ...


class BaseExtension:
    name = "base"

    def on_session_start(self, context: ExtensionContext) -> None:
        return None

    def inject_context(self, context: ExtensionContext) -> list[Message]:
        return []

    def tool_definitions(self, context: ExtensionContext) -> list[ToolDefinition]:
        return []

    def before_llm_call(
        self,
        context: ExtensionContext,
        *,
        step: int,
        messages: list[Message],
        tools: list[ToolDefinition],
    ) -> tuple[list[Message], list[ToolDefinition]]:
        return messages, tools

    def after_llm_call(
        self,
        context: ExtensionContext,
        *,
        step: int,
        reply: Any,
    ) -> None:
        return None

    def on_context_compact(
        self,
        context: ExtensionContext,
        *,
        step: int,
        original_characters: int,
        sent_characters: int,
        compacted_messages: int,
    ) -> None:
        return None

    def before_tool_call(
        self,
        context: ExtensionContext,
        *,
        step: int,
        call: Any,
    ) -> Any | None:
        return call

    def execute_tool(
        self,
        context: ExtensionContext,
        *,
        step: int,
        call: Any,
    ) -> ToolResult | None:
        return None

    def after_tool_call(
        self,
        context: ExtensionContext,
        *,
        step: int,
        call: Any,
        result: ToolResult,
    ) -> ToolResult:
        return result

    def on_session_end(
        self,
        context: ExtensionContext,
        *,
        answer: str | None,
        error: Exception | None,
        state: Any,
    ) -> None:
        return None


class ExtensionManager:
    def __init__(self, extensions: Sequence[AgentExtension] = ()) -> None:
        self.extensions = list(extensions)

    def names(self) -> list[str]:
        return [extension.name for extension in self.extensions]

    def on_session_start(self, context: ExtensionContext) -> None:
        for extension in self.extensions:
            extension.on_session_start(context)

    def inject_context(self, context: ExtensionContext) -> list[Message]:
        messages: list[Message] = []
        for extension in self.extensions:
            messages.extend(extension.inject_context(context))
        return messages

    def tool_definitions(self, context: ExtensionContext) -> list[ToolDefinition]:
        definitions: list[ToolDefinition] = []
        seen: set[str] = set()
        for extension in self.extensions:
            for definition in extension.tool_definitions(context):
                name = _tool_name(definition)
                if name and name not in seen:
                    definitions.append(definition)
                    seen.add(name)
        return definitions

    def before_llm_call(
        self,
        context: ExtensionContext,
        *,
        step: int,
        messages: list[Message],
        tools: list[ToolDefinition],
    ) -> tuple[list[Message], list[ToolDefinition]]:
        for extension in self.extensions:
            original_characters = _messages_characters(messages)
            original_count = len(messages)
            messages, tools = extension.before_llm_call(
                context,
                step=step,
                messages=messages,
                tools=tools,
            )
            sent_characters = _messages_characters(messages)
            compacted_messages = max(0, original_count - len(messages))
            if sent_characters < original_characters:
                self.on_context_compact(
                    context,
                    step=step,
                    original_characters=original_characters,
                    sent_characters=sent_characters,
                    compacted_messages=compacted_messages,
                )
        return messages, tools

    def after_llm_call(
        self,
        context: ExtensionContext,
        *,
        step: int,
        reply: Any,
    ) -> None:
        for extension in self.extensions:
            extension.after_llm_call(context, step=step, reply=reply)

    def on_context_compact(
        self,
        context: ExtensionContext,
        *,
        step: int,
        original_characters: int,
        sent_characters: int,
        compacted_messages: int,
    ) -> None:
        for extension in self.extensions:
            extension.on_context_compact(
                context,
                step=step,
                original_characters=original_characters,
                sent_characters=sent_characters,
                compacted_messages=compacted_messages,
            )

    def before_tool_call(
        self,
        context: ExtensionContext,
        *,
        step: int,
        call: Any,
    ) -> Any | None:
        for extension in self.extensions:
            call = extension.before_tool_call(context, step=step, call=call)
            if call is None:
                return None
        return call

    def execute_tool(
        self,
        context: ExtensionContext,
        *,
        step: int,
        call: Any,
    ) -> ToolResult | None:
        for extension in self.extensions:
            result = extension.execute_tool(context, step=step, call=call)
            if result is not None:
                return result
        return None

    def after_tool_call(
        self,
        context: ExtensionContext,
        *,
        step: int,
        call: Any,
        result: ToolResult,
    ) -> ToolResult:
        for extension in self.extensions:
            result = extension.after_tool_call(
                context,
                step=step,
                call=call,
                result=result,
            )
        return result

    def on_session_end(
        self,
        context: ExtensionContext,
        *,
        answer: str | None,
        error: Exception | None,
        state: Any,
    ) -> None:
        for extension in self.extensions:
            extension.on_session_end(
                context,
                answer=answer,
                error=error,
                state=state,
            )


class ProjectMemoryExtension(BaseExtension):
    name = "project-memory"

    def __init__(self, store: ProjectMemoryStore) -> None:
        self.store = store

    def inject_context(self, context: ExtensionContext) -> list[Message]:
        message = self.store.build_message(context.workspace)
        return [message] if message is not None else []

    def on_session_end(
        self,
        context: ExtensionContext,
        *,
        answer: str | None,
        error: Exception | None,
        state: Any,
    ) -> None:
        if error is not None or answer is None:
            return
        self.store.update_after_task(
            context.workspace,
            task=context.task,
            answer=answer,
            modified_files=tuple(getattr(state, "modified_files", ())),
            latest_command=str(getattr(state, "latest_command", "not run")),
        )
        context.trace.record("project_memory_updated")


class SkillSelectionExtension(BaseExtension):
    name = "skill-selection"

    def __init__(self, registry: SkillRegistry | None = None) -> None:
        self.registry = registry or SkillRegistry.load_builtin()

    def inject_context(self, context: ExtensionContext) -> list[Message]:
        selected = self.registry.select(context.task)
        if not selected:
            return []
        names = [skill.name for skill in selected]
        context.trace.record("skills_selected", skills=names)
        context.emit(f"[技能] 已启用：{', '.join(names)}")
        return [skill.message() for skill in selected]


class SubAgentExtension(BaseExtension):
    name = "subagents"
    DELEGATE_TOOL = "delegate_subagent"
    BASE_TOOLS = {"list_files", "read_file", "search_text"}
    TEST_TOOLS = {"run_command"}
    ROLES = {"researcher", "tester", "reviewer"}

    def __init__(
        self,
        model: Any | None = None,
        *,
        max_steps: int = 4,
        max_tool_characters: int = 8_000,
        complexity_file_threshold: int = 25,
    ) -> None:
        self.model = model
        self.max_steps = max_steps
        self.max_tool_characters = max_tool_characters
        self.complexity_file_threshold = complexity_file_threshold
        self._enabled = False
        self._reason = ""
        self._recommended_focuses: list[str] = []
        self._change_revision = 0
        self._reviewed_revision = 0

    def on_session_start(self, context: ExtensionContext) -> None:
        self._enabled, self._reason = _should_enable_subagents(
            context.task,
            context.workspace,
            file_threshold=self.complexity_file_threshold,
        )
        self._recommended_focuses = _recommended_subagent_focuses(
            context.task,
            context.workspace,
        )
        self._change_revision = 0
        self._reviewed_revision = 0

    def before_llm_call(
        self,
        context: ExtensionContext,
        *,
        step: int,
        messages: list[Message],
        tools: list[ToolDefinition],
    ) -> tuple[list[Message], list[ToolDefinition]]:
        if not self._review_required:
            return messages, tools
        gate = {
            "role": "system",
            "content": (
                "<subagent_review_gate>\n"
                "A complex task changed project files. Before accepting verification, "
                "delegate a reviewer subagent with role='reviewer' to inspect the "
                "current patch or changed files. The reviewer is read/test only and "
                "does not decide whether to ship. The main agent must judge the "
                "evidence, make any needed fixes, then run final verification itself. "
                "If you call run_command before reviewer review is complete, the "
                "framework will convert that request into a reviewer delegation first.\n"
                "</subagent_review_gate>"
            ),
        }
        context.trace.record(
            "subagent_review_required",
            step=step,
            change_revision=self._change_revision,
        )
        return [*messages, gate], tools

    @property
    def _review_required(self) -> bool:
        return self._enabled and self._change_revision > self._reviewed_revision

    def inject_context(self, context: ExtensionContext) -> list[Message]:
        if not self._enabled:
            return []
        focuses = self._recommended_focuses or ["general"]
        context.trace.record(
            "subagent_policy_enabled",
            reason=self._reason,
            recommended_focuses=focuses,
        )
        context.emit(
            f"[SubAgent] 已启用委派策略：{self._reason}；建议关注 {', '.join(focuses)}"
        )
        return [
            {
                "role": "system",
                "content": (
                    "<subagent_policy>\n"
                    "Three bounded SubAgent roles are available because this task or workspace "
                    "appears complex. The main agent must decide whether delegation actually "
                    "reduces context load or improves quality. Do not delegate simple single-file work.\n"
                    f"Recommended delegation focuses for this task: {', '.join(focuses)}.\n"
                    "Use role='researcher' to locate implementations, related modules, existing "
                    "patterns, architecture constraints, and risks. Researcher is read/search only.\n"
                    "Use role='tester' to design a verification strategy, identify missing coverage, "
                    "run allowed tests, and explain failure logs. Tester cannot edit files.\n"
                    "Use role='reviewer' after non-trivial changes to inspect the patch, evidence, "
                    "risks, and next verification step. Reviewer can read/search and run allowed "
                    "tests, but cannot edit files.\n"
                    "Subagents cannot create other subagents and never make the final decision. "
                    "The main agent remains responsible for judging reports, applying fixes, "
                    "running final verification, and producing the user-facing answer.\n"
                    "</subagent_policy>"
                ),
            }
        ]

    def tool_definitions(self, context: ExtensionContext) -> list[ToolDefinition]:
        if not self._enabled or self.model is None:
            return []
        return [
            {
                "type": "function",
                "function": {
                    "name": self.DELEGATE_TOOL,
                    "description": (
                        "Delegate one bounded subagent with its own context. Use researcher "
                        "for read-only architecture or module discovery, tester for validation "
                        "strategy and allowed test execution, and reviewer for patch review. "
                        "Use only when context separation is worth the cost."
                    ),
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "role": {
                                "type": "string",
                                "enum": ["researcher", "tester", "reviewer"],
                                "description": (
                                    "researcher reads/searches architecture; tester plans/runs "
                                    "allowed tests; reviewer inspects changes before final verification."
                                ),
                            },
                            "question": {
                                "type": "string",
                                "description": "The specific question this subagent should answer.",
                            },
                            "focus": {
                                "type": "string",
                                "description": "Short scope label for this delegation.",
                            },
                            "paths": {
                                "type": "array",
                                "items": {"type": "string"},
                                "description": "Optional workspace-relative paths to prioritize.",
                            },
                            "search_terms": {
                                "type": "array",
                                "items": {"type": "string"},
                                "description": "Optional terms the subagent should search for.",
                            },
                            "constraints": {
                                "type": "array",
                                "items": {"type": "string"},
                                "description": "Constraints the subagent must obey.",
                            },
                            "expected_output": {
                                "type": "string",
                                "description": "Expected structured result shape.",
                            },
                        },
                        "required": ["role", "question"],
                        "additionalProperties": False,
                    },
                },
            }
        ]

    def execute_tool(
        self,
        context: ExtensionContext,
        *,
        step: int,
        call: Any,
    ) -> ToolResult | None:
        if call.name != self.DELEGATE_TOOL:
            return None
        if not self._enabled or self.model is None:
            return ToolResult(
                "Subagent delegation is not enabled for this task.", success=False
            )
        role = _normalize_subagent_role(call.arguments.get("role"))
        if role not in self.ROLES:
            return ToolResult(f"Unsupported subagent role: {role}", success=False)
        question = str(call.arguments.get("question", "")).strip()
        focus = str(call.arguments.get("focus", role)).strip() or role
        paths = _string_sequence(call.arguments.get("paths"))
        search_terms = _string_sequence(call.arguments.get("search_terms"))
        constraints = _string_sequence(call.arguments.get("constraints"))
        expected_output = str(
            call.arguments.get(
                "expected_output",
                _default_subagent_expected_output(role),
            )
        )
        task = SubagentTask(
            id=str(getattr(call, "id", "")) or _subagent_task_id(role, question),
            role=role,
            objective=question,
            context=tuple([*paths, *search_terms]),
            constraints=tuple(constraints),
            allowed_tools=tuple(sorted(self._allowed_tools_for_role(role))),
            expected_output=expected_output,
            status="running",
        )
        context.emit(f"[SubAgent] {role}：{_compact_inline(question, 120)}")
        context.trace.record(
            "subagent_start",
            step=step,
            task=_subagent_task_data(task),
        )
        report = self._run_subagent(
            context,
            step=step,
            task=task,
            focus=focus,
        )
        if role == "reviewer":
            self._reviewed_revision = self._change_revision
            context.emit(_reviewer_result_event(report))
            context.trace.record(
                "subagent_review_completed",
                step=step,
                change_revision=self._change_revision,
            )
        context.trace.record("subagent_complete", step=step, role=role)
        return ToolResult(report, success=True)

    def before_tool_call(
        self,
        context: ExtensionContext,
        *,
        step: int,
        call: Any,
    ) -> Any | None:
        if self._review_required and getattr(call, "name", "") == "run_command":
            argv = getattr(call, "arguments", {}).get("argv", [])
            command = " ".join(str(part) for part in argv) if isinstance(argv, list) else str(argv)
            context.trace.record(
                "subagent_review_auto_delegated",
                step=step,
                change_revision=self._change_revision,
                attempted_tool="run_command",
                attempted_command=_compact_inline(command, 180),
            )
            return SimpleNamespace(
                id=getattr(call, "id", "review-before-verification"),
                name=self.DELEGATE_TOOL,
                arguments={
                    "role": "reviewer",
                    "question": (
                        "Review the changed files before final verification. "
                        f"The main agent attempted to run: {command or 'run_command'}"
                    ),
                    "focus": "pre-verification review",
                    "constraints": [
                        "Read and test only; do not edit files.",
                        "Report evidence, risks, confidence, and the next verification step.",
                    ],
                },
            )
        return call

    def after_tool_call(
        self,
        context: ExtensionContext,
        *,
        step: int,
        call: Any,
        result: ToolResult,
    ) -> ToolResult:
        if (
            self._enabled
            and result.success
            and getattr(call, "name", "") in {"write_file", "apply_patch"}
        ):
            self._change_revision += 1
            context.trace.record(
                "subagent_review_pending",
                step=step,
                change_revision=self._change_revision,
                tool=getattr(call, "name", "unknown"),
            )
        return result

    def _run_subagent(
        self,
        context: ExtensionContext,
        *,
        step: int,
        task: SubagentTask,
        focus: str,
    ) -> str:
        registry = ToolRegistry(context.workspace)
        allowed_tool_names = self._allowed_tools_for_role(task.role)
        subagent_tools = [
            definition
            for definition in registry.definitions
            if _tool_name(definition) in allowed_tool_names
        ]
        messages: list[Message] = [
            {
                "role": "system",
                "content": (
                    "You are a read-only subagent. You have an independent context "
                    "for one focused investigation. Use only the provided tools. "
                    + _subagent_role_instructions(task.role)
                    + " Return one JSON object with taskId, summary, findings, "
                    "artifacts, confidence, risks, and recommendedNextStep. Do not ask "
                    "the user questions."
                ),
            },
            {
                "role": "user",
                "content": _subagent_assignment(
                    task=task,
                    focus=focus,
                ),
            },
        ]
        last_content = ""
        for substep in range(1, self.max_steps + 1):
            try:
                reply = self.model.complete(messages, subagent_tools)
            except Exception as error:
                context.trace.record(
                    "subagent_error",
                    step=step,
                    substep=substep,
                    role=task.role,
                    error_type=type(error).__name__,
                )
                return _subagent_report(
                    task=task,
                    focus=focus,
                    status="failed",
                    content=f"Subagent failed before producing a report: {error}",
                )
            last_content = str(getattr(reply, "content", "") or last_content)
            tool_calls = tuple(getattr(reply, "tool_calls", ()) or ())
            context.trace.record(
                "subagent_model_reply",
                step=step,
                substep=substep,
                role=task.role,
                tools=[getattr(tool_call, "name", "unknown") for tool_call in tool_calls],
            )
            if not tool_calls:
                return _subagent_report(
                    task=task,
                    focus=focus,
                    status="complete",
                    content=last_content or "No findings reported.",
                )
            messages.append(_assistant_tool_message(reply))
            for tool_call in tool_calls:
                tool_name = str(getattr(tool_call, "name", "unknown"))
                context.emit(f"[SubAgent:{task.role}] {tool_name}")
                if tool_name not in allowed_tool_names:
                    result = f"ERROR: {task.role} subagents may not call {tool_name}"
                    success = False
                else:
                    try:
                        result = registry.execute(
                            tool_name,
                            getattr(tool_call, "arguments", {}),
                        )
                        success = True
                    except (OSError, ValueError) as error:
                        result = f"ERROR: {error}"
                        success = False
                context.trace.record(
                    "subagent_tool_result",
                    step=step,
                    substep=substep,
                    role=task.role,
                    tool=tool_name,
                    success=success,
                )
                messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": str(getattr(tool_call, "id", "")),
                        "content": _limit_text(result, self.max_tool_characters),
                    }
                )
        return _subagent_report(
            task=task,
            focus=focus,
            status="max_steps_reached",
            content=last_content or "Subagent reached its step budget without a final report.",
        )

    def _allowed_tools_for_role(self, role: str) -> set[str]:
        if role == "researcher":
            return set(self.BASE_TOOLS)
        return {*self.BASE_TOOLS, *self.TEST_TOOLS}


class ContextPackExtension(BaseExtension):
    name = "context-pack"
    READ_OUTPUT_TOOL = "read_context_output"

    def __init__(
        self,
        *,
        max_characters: int = 40_000,
        recent_tool_exchanges: int = 3,
        summary_characters: int = 2_400,
        offload_characters: int = 12_000,
        offload_preview_characters: int = 2_400,
        output_read_characters: int = 12_000,
        output_store: Path | None = None,
    ) -> None:
        self.max_characters = max_characters
        self.recent_tool_exchanges = recent_tool_exchanges
        self.summary_characters = summary_characters
        self.offload_characters = offload_characters
        self.offload_preview_characters = offload_preview_characters
        self.output_read_characters = output_read_characters
        self.output_store = output_store
        self._store_dir: Path | None = None
        self._records: dict[str, ContextOutputRecord] = {}

    def on_session_start(self, context: ExtensionContext) -> None:
        self._store_dir = self._resolve_store_dir(context)
        self._store_dir.mkdir(parents=True, exist_ok=True)
        self._records = self._load_records(self._store_dir)

    def tool_definitions(self, context: ExtensionContext) -> list[ToolDefinition]:
        return [
            {
                "type": "function",
                "function": {
                    "name": self.READ_OUTPUT_TOOL,
                    "description": (
                        "Read exact text from a large tool output that was offloaded "
                        "from the model context. Use this when an offloaded result "
                        "summary shows a real output_id. Never invent ids; if no "
                        "output_id is visible, rerun or inspect the source tool instead."
                    ),
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "output_id": {
                                "type": "string",
                                "description": "The output_id shown in an offloaded tool result.",
                            },
                            "start_line": {
                                "type": "integer",
                                "minimum": 1,
                                "description": "Optional first line to read, starting at 1.",
                            },
                            "end_line": {
                                "type": "integer",
                                "minimum": 1,
                                "description": "Optional final line to read, inclusive.",
                            },
                            "max_characters": {
                                "type": "integer",
                                "minimum": 500,
                                "maximum": 30000,
                                "description": "Maximum characters to return.",
                            },
                        },
                        "required": ["output_id"],
                        "additionalProperties": False,
                    },
                },
            }
        ]

    def execute_tool(
        self,
        context: ExtensionContext,
        *,
        step: int,
        call: Any,
    ) -> ToolResult | None:
        if call.name != self.READ_OUTPUT_TOOL:
            return None
        store_dir = self._ensure_store_dir(context)
        output_id = str(call.arguments.get("output_id", "")).strip()
        record = self._records.get(output_id)
        if record is None:
            self._records = self._load_records(store_dir)
            record = self._records.get(output_id)
        if record is None:
            return ToolResult(
                (
                    f"ERROR: unknown context output id: {output_id}. "
                    "Only call read_context_output with an output_id copied from "
                    "a visible <offloaded_tool_result>; rerun the original command "
                    "or inspect files when no id is available."
                ),
                False,
            )
        path = self._record_path(store_dir, record)
        if path is None or not path.is_file():
            return ToolResult(f"ERROR: context output is missing: {output_id}", False)
        text = path.read_text(encoding="utf-8", errors="replace")
        start_line = _positive_int(call.arguments.get("start_line"), 1)
        end_line = _positive_int(call.arguments.get("end_line"), 0)
        max_characters = _positive_int(
            call.arguments.get("max_characters"),
            self.output_read_characters,
        )
        max_characters = min(max(max_characters, 500), 30_000)
        lines = text.splitlines()
        start_index = max(start_line - 1, 0)
        end_index = end_line if end_line else len(lines)
        selected = lines[start_index:end_index]
        numbered = [
            f"{line_number}: {line}"
            for line_number, line in enumerate(selected, start=start_index + 1)
        ]
        body = "\n".join(numbered)
        truncated = len(body) > max_characters
        body = _limit_text(body, max_characters)
        header = [
            f"<context_output id=\"{record.output_id}\" tool=\"{record.tool}\">",
            f"Lines returned: {start_index + 1}-{min(end_index, len(lines))} of {len(lines)}",
        ]
        if truncated:
            header.append(
                "Output truncated. Request a narrower start_line/end_line range for more."
            )
        return ToolResult("\n".join([*header, body, "</context_output>"]))

    def after_tool_call(
        self,
        context: ExtensionContext,
        *,
        step: int,
        call: Any,
        result: ToolResult,
    ) -> ToolResult:
        if call.name == self.READ_OUTPUT_TOOL:
            return result
        if len(result.content) <= self.offload_characters:
            return result
        record = self._offload_tool_result(context, step=step, call=call, result=result)
        context.emit(
            f"[上下文] 已外置长工具输出：{call.name} -> {record.output_id}"
        )
        context.trace.record(
            "context_output_offloaded",
            step=step,
            tool=call.name,
            output_id=record.output_id,
            characters=record.characters,
            lines=record.lines,
            success=result.success,
        )
        return ToolResult(
            self._offloaded_result_message(call=call, record=record, result=result),
            result.success,
        )

    def before_llm_call(
        self,
        context: ExtensionContext,
        *,
        step: int,
        messages: list[Message],
        tools: list[ToolDefinition],
    ) -> tuple[list[Message], list[ToolDefinition]]:
        original_characters = _messages_characters(messages)
        if original_characters <= self.max_characters:
            context.trace.record(
                "context_pack_built",
                step=step,
                original_characters=original_characters,
                sent_characters=original_characters,
                compacted_messages=0,
                mode="passthrough",
            )
            return messages, tools

        packed, compacted = self._compact_tool_exchanges(messages)
        sent_characters = _messages_characters(packed)
        if sent_characters > self.max_characters:
            packed = self._trim_old_plain_messages(packed)
            sent_characters = _messages_characters(packed)

        if compacted:
            context.emit(
                f"[上下文] 已打包 {compacted} 条旧工具消息，"
                f"请求字符 {original_characters}->{sent_characters}"
            )
        context.trace.record(
            "context_pack_built",
            step=step,
            original_characters=original_characters,
            sent_characters=sent_characters,
            compacted_messages=compacted,
            mode="compact" if compacted else "trim",
        )
        return packed, tools

    def _compact_tool_exchanges(self, messages: list[Message]) -> tuple[list[Message], int]:
        exchanges = _tool_exchange_spans(messages)
        if len(exchanges) <= self.recent_tool_exchanges:
            return list(messages), 0
        compact_spans = set(exchanges[: -self.recent_tool_exchanges])
        packed: list[Message] = []
        compacted_messages = 0
        index = 0
        while index < len(messages):
            span = next((span for span in compact_spans if span[0] == index), None)
            if span is None:
                packed.append(messages[index])
                index += 1
                continue
            start, end = span
            packed.append(
                {
                    "role": "system",
                    "content": self._summarize_exchange(messages[start : end + 1]),
                }
            )
            compacted_messages += end - start + 1
            index = end + 1
        return packed, compacted_messages

    def _summarize_exchange(self, exchange: list[Message]) -> str:
        assistant = exchange[0]
        tool_calls = assistant.get("tool_calls", [])
        call_by_id = {
            str(call.get("id", "")): call
            for call in tool_calls
            if isinstance(call, dict)
        }
        checked: list[str] = []
        modified: list[str] = []
        commands: list[str] = []
        findings: list[str] = []
        risks: list[str] = []
        next_steps: list[str] = []
        for message in exchange[1:]:
            call_id = str(message.get("tool_call_id", ""))
            call = call_by_id.get(call_id, {})
            name, arguments = _call_identity(call)
            content = str(message.get("content", ""))
            result_summary = _summarize_tool_result(name, arguments, content)
            if check := _checked_summary(name, arguments):
                checked.append(check)
            if change := _modified_summary(name, arguments):
                modified.append(change)
            if command := _command_summary(name, arguments):
                commands.append(command)
            findings.append(f"{name}: {result_summary}")
            if _looks_like_failure(content):
                risks.append(f"{name} returned an error or failing check.")
                next_steps.append("Inspect the failing output and repair before final delivery.")
            elif name in {"write_file", "apply_patch"}:
                next_steps.append("Run or keep the relevant verification after this change.")
            elif name in {"read_file", "search_text", "list_files"}:
                next_steps.append("Use live file reads for exact details before editing.")
        lines = [
            "<compacted_history>",
            "Older work summarized for context budget. Exact long outputs may be "
            "available via read_context_output only when an explicit output_id is "
            "listed in a visible <offloaded_tool_result>. Never invent ids.",
            "Checked:",
            *_bullet_lines(checked),
            "Modified:",
            *_bullet_lines(modified),
            "Commands:",
            *_bullet_lines(commands),
            "Findings:",
            *_bullet_lines(findings),
            "Risks:",
            *_bullet_lines(_unique(risks)),
            "Next:",
            *_bullet_lines(_unique(next_steps)),
        ]
        lines.append("</compacted_history>")
        return _limit_text("\n".join(lines), self.summary_characters)

    def _trim_old_plain_messages(self, messages: list[Message]) -> list[Message]:
        if _messages_characters(messages) <= self.max_characters:
            return messages
        packed: list[Message] = []
        last_user_index = max(
            (index for index, message in enumerate(messages) if message.get("role") == "user"),
            default=len(messages) - 1,
        )
        for index, message in enumerate(messages):
            content = message.get("content")
            if (
                isinstance(content, str)
                and index < last_user_index
                and message.get("role") in {"assistant", "tool"}
                and len(content) > 1_200
            ):
                updated = dict(message)
                updated["content"] = _limit_text(content, 1_200)
                packed.append(updated)
            else:
                packed.append(message)
        return packed

    def _resolve_store_dir(self, context: ExtensionContext) -> Path:
        workspace_key = hashlib.sha256(
            str(context.workspace.root).lower().encode("utf-8", errors="replace")
        ).hexdigest()[:16]
        base = self.output_store or Path.cwd() / ".coding-agent" / "context-outputs"
        return (base / workspace_key).resolve()

    def _ensure_store_dir(self, context: ExtensionContext) -> Path:
        if self._store_dir is None:
            self.on_session_start(context)
        assert self._store_dir is not None
        return self._store_dir

    def _index_path(self, store_dir: Path) -> Path:
        return store_dir / "index.json"

    def _load_records(self, store_dir: Path) -> dict[str, ContextOutputRecord]:
        index_path = self._index_path(store_dir)
        if not index_path.is_file():
            return {}
        try:
            raw = json.loads(index_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return {}
        if not isinstance(raw, dict):
            return {}
        records: dict[str, ContextOutputRecord] = {}
        for output_id, value in raw.items():
            if not isinstance(output_id, str) or not isinstance(value, dict):
                continue
            try:
                records[output_id] = ContextOutputRecord(
                    output_id=output_id,
                    filename=str(value["filename"]),
                    tool=str(value.get("tool", "unknown")),
                    step=int(value.get("step", 0)),
                    characters=int(value.get("characters", 0)),
                    lines=int(value.get("lines", 0)),
                    success=bool(value.get("success", True)),
                    summary=str(value.get("summary", "")),
                )
            except (KeyError, TypeError, ValueError):
                continue
        return records

    def _save_records(self, store_dir: Path) -> None:
        payload = {
            output_id: {
                "filename": record.filename,
                "tool": record.tool,
                "step": record.step,
                "characters": record.characters,
                "lines": record.lines,
                "success": record.success,
                "summary": record.summary,
            }
            for output_id, record in sorted(self._records.items())
        }
        self._index_path(store_dir).write_text(
            json.dumps(payload, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    def _offload_tool_result(
        self,
        context: ExtensionContext,
        *,
        step: int,
        call: Any,
        result: ToolResult,
    ) -> ContextOutputRecord:
        store_dir = self._ensure_store_dir(context)
        arguments = getattr(call, "arguments", {})
        digest = hashlib.sha256(result.content.encode("utf-8", errors="replace")).hexdigest()
        tool_name = str(getattr(call, "name", "tool"))
        slug = _safe_slug(tool_name)
        output_id = f"context-output-step{step}-{slug}-{digest[:10]}"
        filename = f"{output_id}.txt"
        (store_dir / filename).write_text(result.content, encoding="utf-8")
        record = ContextOutputRecord(
            output_id=output_id,
            filename=filename,
            tool=tool_name,
            step=step,
            characters=len(result.content),
            lines=len(result.content.splitlines()),
            success=result.success,
            summary=_summarize_tool_result(tool_name, arguments, result.content),
        )
        self._records[output_id] = record
        self._save_records(store_dir)
        return record

    def _record_path(
        self,
        store_dir: Path,
        record: ContextOutputRecord,
    ) -> Path | None:
        path = (store_dir / record.filename).resolve()
        try:
            path.relative_to(store_dir.resolve())
        except ValueError:
            return None
        return path

    def _offloaded_result_message(
        self,
        *,
        call: Any,
        record: ContextOutputRecord,
        result: ToolResult,
    ) -> str:
        arguments = getattr(call, "arguments", {})
        preview = _sample_lines(result.content, 18)
        return "\n".join(
            [
                (
                    f"<offloaded_tool_result output_id=\"{record.output_id}\" "
                    f"tool=\"{record.tool}\" characters=\"{record.characters}\" "
                    f"lines=\"{record.lines}\" success=\"{str(record.success).lower()}\">"
                ),
                "Full output was stored outside the model context.",
                (
                    "Use read_context_output with this output_id and a narrow "
                    "line range when exact text is needed."
                ),
                f"Args: {_compact_inline(json.dumps(arguments, ensure_ascii=False), 420)}",
                f"Key summary: {record.summary}",
                f"Preview: {_limit_text(preview, self.offload_preview_characters)}",
                "</offloaded_tool_result>",
            ]
        )


def _tool_name(definition: ToolDefinition) -> str:
    function = definition.get("function")
    if not isinstance(function, dict):
        return ""
    name = function.get("name")
    return name if isinstance(name, str) else ""


def _should_enable_subagents(
    task: str,
    workspace: Workspace,
    *,
    file_threshold: int,
) -> tuple[bool, str]:
    normalized = task.casefold()
    explicit = any(
        marker in normalized
        for marker in ("subagent", "sub-agent", "子agent", "子 agent", "多代理")
    )
    if explicit:
        return True, "用户明确提到 SubAgent"

    profile = _workspace_profile(workspace, limit=max(file_threshold + 10, 80))
    score, reasons = _requirement_complexity_score(
        task,
        profile=profile,
        file_threshold=file_threshold,
    )
    if score >= 5:
        return True, f"需求复杂度评分 {score}：{'；'.join(reasons)}"
    return False, f"需求复杂度评分 {score}，无需委派：{'；'.join(reasons) or '任务范围较小'}"


def _requirement_complexity_score(
    task: str,
    *,
    profile: dict[str, object],
    file_threshold: int,
) -> tuple[int, list[str]]:
    normalized = task.casefold()
    reasons: list[str] = []
    score = 0

    legacy_markers = (
        "大型",
        "复杂",
        "多模块",
        "全局",
        "全面分析",
        "架构",
        "运行失败",
        "为什么失败",
        "性能",
        "安全",
        "前端",
        "后端",
        "数据库",
        "large",
        "complex",
        "monorepo",
        "architecture",
        "frontend",
        "backend",
        "database",
        "performance",
        "security",
    )
    legacy_score = sum(1 for marker in legacy_markers if marker in normalized)
    if legacy_score:
        score += min(legacy_score, 4)
        reasons.append(f"显式复杂度信号 {legacy_score} 个")

    feature_count = _feature_count(task)
    if feature_count >= 7:
        score += 4
        reasons.append(f"功能点约 {feature_count} 个")
    elif feature_count >= 4:
        score += 3
        reasons.append(f"功能点约 {feature_count} 个")
    elif feature_count >= 2:
        score += 1
        reasons.append(f"功能点约 {feature_count} 个")

    delivery_markers = (
        "系统",
        "平台",
        "管理后台",
        "后台管理",
        "完整项目",
        "完整应用",
        "桌面应用",
        "桌面小应用",
        "网站",
        "web app",
        "api 服务",
        "api service",
        "from scratch",
        "从零",
        "搭建",
        "开发一个",
        "创建一个",
    )
    if any(marker in normalized for marker in delivery_markers):
        score += 2
        reasons.append("交付形态是完整系统/应用")

    task_domains = _task_domains(normalized)
    workspace_domains = {str(domain) for domain in profile["domains"]}
    domains = task_domains | workspace_domains
    domain_score = len(domains)
    file_count = int(profile["file_count"])
    if domain_score >= 3:
        score += 3
        reasons.append(f"涉及 {domain_score} 个领域")
    elif domain_score >= 2:
        score += 2
        reasons.append(f"涉及 {domain_score} 个领域")

    if file_count >= file_threshold * 2:
        score += 3
        reasons.append(f"工作区约 {file_count} 个文件")
    elif file_count >= file_threshold:
        score += 2
        reasons.append(f"工作区约 {file_count} 个文件")

    quality_markers = (
        "失败",
        "报错",
        "修复",
        "重构",
        "迁移",
        "兼容",
        "权限",
        "登录",
        "认证",
        "鉴权",
        "性能",
        "安全",
        "测试覆盖",
        "test coverage",
        "failing",
        "error",
        "refactor",
        "migration",
        "auth",
        "permission",
    )
    quality_score = sum(1 for marker in quality_markers if marker in normalized)
    if quality_score >= 2:
        score += 2
        reasons.append(f"质量/风险信号 {quality_score} 个")
    elif quality_score == 1:
        score += 1
        reasons.append("存在质量/风险信号")

    return score, reasons


def _feature_count(task: str) -> int:
    normalized = task.casefold()
    feature_markers = (
        "登录",
        "注册",
        "权限",
        "认证",
        "鉴权",
        "用户",
        "角色",
        "管理",
        "搜索",
        "筛选",
        "排序",
        "统计",
        "报表",
        "导出",
        "导入",
        "上传",
        "下载",
        "通知",
        "配置",
        "设置",
        "支付",
        "订单",
        "题库",
        "考试",
        "发布",
        "判分",
        "成绩",
        "学生端",
        "教师端",
        "后台",
        "接口",
        "数据库",
        "缓存",
        "部署",
        "login",
        "register",
        "auth",
        "permission",
        "role",
        "search",
        "filter",
        "report",
        "export",
        "upload",
        "notification",
        "settings",
        "payment",
        "database",
        "deploy",
    )
    marker_count = sum(1 for marker in feature_markers if marker in normalized)
    list_count = 0
    if any(marker in normalized for marker in ("包含", "含有", "支持", "需要", "具备", "实现")):
        parts = [
            part.strip()
            for part in re.split(r"[、,，;；\n]+|以及|并且|和", task)
            if len(part.strip()) >= 2
        ]
        list_count = max(0, len(parts) - 1)
    return max(marker_count, list_count)


def _task_domains(normalized: str) -> set[str]:
    domains: set[str] = set()
    domain_markers = {
        "frontend": ("前端", "页面", "界面", "ui", "web", "react", "vue", "client"),
        "backend": ("后端", "接口", "api", "server", "service", "服务端"),
        "database": ("数据库", "db", "schema", "migration", "存储", "持久化"),
        "tests": ("测试", "验证", "pytest", "unittest", "test", "spec"),
        "desktop": ("桌面", "tkinter", "gui", "windows"),
    }
    for domain, markers in domain_markers.items():
        if any(marker in normalized for marker in markers):
            domains.add(domain)
    return domains


def _workspace_profile(workspace: Workspace, *, limit: int) -> dict[str, object]:
    protected = {".git", ".venv", ".coding-agent", "__pycache__"}
    domains: set[str] = set()
    file_count = 0
    try:
        iterator = workspace.root.rglob("*")
        for path in iterator:
            if any(part in protected for part in path.relative_to(workspace.root).parts):
                continue
            if not path.is_file():
                continue
            file_count += 1
            relative = path.relative_to(workspace.root).as_posix().casefold()
            if any(part in relative for part in ("frontend", "client", "web", "ui")):
                domains.add("frontend")
            if any(part in relative for part in ("backend", "server", "api", "service")):
                domains.add("backend")
            if any(part in relative for part in ("db", "database", "migration", "schema")):
                domains.add("database")
            if any(part in relative for part in ("test", "spec")):
                domains.add("tests")
            if file_count >= limit:
                break
    except OSError:
        return {"file_count": file_count, "domains": sorted(domains)}
    return {"file_count": file_count, "domains": sorted(domains)}


def _recommended_subagent_focuses(task: str, workspace: Workspace) -> list[str]:
    normalized = task.casefold()
    profile = _workspace_profile(workspace, limit=120)
    roles: list[str] = []
    domain_to_role = {
        "frontend": "frontend",
        "backend": "backend",
        "database": "database",
        "tests": "tests",
    }
    for domain in profile["domains"]:
        role = domain_to_role.get(str(domain))
        if role is not None:
            roles.append(role)
    keyword_roles = (
        ("frontend", ("前端", "frontend", "ui", "web")),
        ("backend", ("后端", "backend", "api", "server", "service")),
        ("database", ("数据库", "database", "db", "schema", "migration")),
        ("tests", ("测试", "test", "pytest", "unittest", "失败")),
        ("performance", ("性能", "performance", "slow", "latency", "速度")),
        ("security", ("安全", "security", "auth", "权限", "登录")),
        ("architecture", ("架构", "architecture", "模块", "依赖")),
    )
    for role, markers in keyword_roles:
        if any(marker in normalized for marker in markers):
            roles.append(role)
    return _unique(roles)[:4]


def _string_sequence(value: object) -> list[str]:
    if not isinstance(value, list):
        return []
    return [_compact_inline(str(item), 160) for item in value if str(item).strip()]


def _normalize_subagent_role(value: object) -> str:
    role = str(value or "reviewer").strip().casefold()
    return role


def _default_subagent_expected_output(role: str) -> str:
    if role == "researcher":
        return (
            "JSON object with summary, findings, artifacts, confidence, risks, "
            "recommendedNextStep. Focus on implementation locations, related modules, "
            "existing patterns, architecture constraints, and risk points."
        )
    if role == "tester":
        return (
            "JSON object with summary, findings, artifacts, confidence, risks, "
            "recommendedNextStep. Focus on verification strategy, missing coverage, "
            "test commands/results, and failure-log interpretation."
        )
    return (
        "JSON object with summary, findings, artifacts, confidence, risks, "
        "recommendedNextStep. Focus on patch correctness, evidence, residual risks, "
        "and the next verification step."
    )


def _subagent_role_instructions(role: str) -> str:
    if role == "researcher":
        return (
            "As researcher, locate where the requested behavior is implemented, "
            "identify related modules and existing project patterns, summarize "
            "architecture constraints, and call out risks. You cannot edit files, "
            "run commands, or make final decisions."
        )
    if role == "tester":
        return (
            "As tester, design the verification strategy, identify missing coverage, "
            "run only allowed read-only test commands, and explain failure logs. "
            "You cannot edit files or make final decisions; propose test additions "
            "as recommendations only."
        )
    return (
        "As reviewer, inspect the current change or proposed patch, evaluate evidence "
        "and residual risks, and recommend the next verification step. You cannot edit "
        "files or make final decisions."
    )


def _subagent_task_id(role: str, objective: str) -> str:
    digest = hashlib.sha1(f"{role}\n{objective}".encode("utf-8")).hexdigest()[:10]
    return f"{role}-{digest}"


def _subagent_task_data(task: SubagentTask) -> dict[str, object]:
    return {
        "id": task.id,
        "role": task.role,
        "objective": task.objective,
        "context": list(task.context),
        "constraints": list(task.constraints),
        "allowedTools": list(task.allowed_tools),
        "expectedOutput": task.expected_output,
        "status": task.status,
    }


def _subagent_assignment(
    *,
    task: SubagentTask,
    focus: str,
) -> str:
    payload = _subagent_task_data(task)
    payload["focus"] = focus
    return (
        "Review this task for the main agent. The main agent may accept or reject "
        "your findings based on evidence quality.\n\n"
        "SubagentTask:\n"
        f"{json.dumps(payload, ensure_ascii=False, indent=2)}\n\n"
        "Return only one JSON object matching SubagentResult:\n"
        "{\n"
        '  "taskId": string,\n'
        '  "summary": string,\n'
        '  "findings": [{"severity": string, "evidence": string, "detail": string}],\n'
        '  "artifacts": [string],\n'
        '  "confidence": number,\n'
        '  "risks": [string],\n'
        '  "recommendedNextStep": string\n'
        "}"
    )


def _subagent_report(
    *,
    task: SubagentTask,
    focus: str,
    status: str,
    content: str,
) -> str:
    result = _subagent_result_data(
        _normalize_subagent_result(task, status=status, content=content)
    )
    return "\n".join(
        [
            (
                f"<subagent_report task_id=\"{task.id}\" role=\"{task.role}\" "
                f"focus=\"{focus}\" status=\"{status}\">"
            ),
            _limit_text(json.dumps(result, ensure_ascii=False, indent=2), 4_000),
            "</subagent_report>",
        ]
    )


def _normalize_subagent_result(
    task: SubagentTask,
    *,
    status: str,
    content: str,
) -> SubagentResult:
    data: dict[str, object] = {}
    try:
        parsed = json.loads(content)
        if isinstance(parsed, dict):
            data = parsed
    except json.JSONDecodeError:
        data = {}
    findings = data.get("findings", ())
    if not isinstance(findings, list):
        findings = []
    normalized_findings: list[dict[str, object]] = [
        item for item in findings if isinstance(item, dict)
    ]
    artifacts = tuple(_string_sequence(data.get("artifacts")))
    risks = tuple(_string_sequence(data.get("risks")))
    confidence = data.get("confidence")
    if isinstance(confidence, (int, float)):
        normalized_confidence = max(0.0, min(float(confidence), 1.0))
    else:
        normalized_confidence = 0.5 if status == "complete" else 0.0
    summary = str(data.get("summary") or content or "No findings reported.").strip()
    recommended_next_step = str(data.get("recommendedNextStep") or "").strip()
    if not recommended_next_step:
        recommended_next_step = "Main agent should judge these findings and run final verification."
    return SubagentResult(
        task_id=str(data.get("taskId") or task.id),
        summary=_limit_text(summary, 1_600),
        findings=tuple(normalized_findings),
        artifacts=artifacts,
        confidence=normalized_confidence,
        risks=risks,
        recommended_next_step=_limit_text(recommended_next_step, 400),
    )


def _subagent_result_data(result: SubagentResult) -> dict[str, object]:
    return {
        "taskId": result.task_id,
        "summary": result.summary,
        "findings": list(result.findings),
        "artifacts": list(result.artifacts),
        "confidence": result.confidence,
        "risks": list(result.risks),
        "recommendedNextStep": result.recommended_next_step,
    }


def _reviewer_result_event(report: str) -> str:
    data = _extract_subagent_report_data(report)
    findings = data.get("findings", [])
    risks = data.get("risks", [])
    if not isinstance(findings, list):
        findings = []
    if not isinstance(risks, list):
        risks = []
    payload = {
        "status": "issues" if findings else "passed",
        "summary": _compact_inline(str(data.get("summary", "Review completed.")), 360),
        "findings": [_compact_finding(item) for item in findings[:5]],
        "risks": [_compact_inline(str(item), 240) for item in risks[:5]],
        "confidence": data.get("confidence", 0.5),
        "recommendedNextStep": _compact_inline(
            str(data.get("recommendedNextStep", "")),
            260,
        ),
    }
    return REVIEWER_RESULT_EVENT_PREFIX + json.dumps(payload, ensure_ascii=False)


def _extract_subagent_report_data(report: str) -> dict[str, object]:
    lines = report.splitlines()
    if len(lines) < 3:
        return {}
    body = "\n".join(lines[1:-1])
    try:
        parsed = json.loads(body)
    except json.JSONDecodeError:
        return {}
    return parsed if isinstance(parsed, dict) else {}


def _compact_finding(item: object) -> dict[str, str]:
    if not isinstance(item, dict):
        return {
            "severity": "info",
            "detail": _compact_inline(str(item), 260),
            "evidence": "",
        }
    return {
        "severity": _compact_inline(str(item.get("severity", "info")), 40),
        "detail": _compact_inline(str(item.get("detail", "")), 260),
        "evidence": _compact_inline(str(item.get("evidence", "")), 180),
    }


def _assistant_tool_message(reply: Any) -> Message:
    return {
        "role": "assistant",
        "content": getattr(reply, "content", None),
        "tool_calls": [
            {
                "id": str(getattr(call, "id", "")),
                "type": "function",
                "function": {
                    "name": str(getattr(call, "name", "unknown")),
                    "arguments": json.dumps(
                        getattr(call, "arguments", {}),
                        ensure_ascii=False,
                    ),
                },
            }
            for call in tuple(getattr(reply, "tool_calls", ()) or ())
        ],
    }


def _messages_characters(messages: list[Message]) -> int:
    return sum(len(json.dumps(message, ensure_ascii=False, default=str)) for message in messages)


def _tool_exchange_spans(messages: list[Message]) -> list[tuple[int, int]]:
    spans: list[tuple[int, int]] = []
    index = 0
    while index < len(messages):
        message = messages[index]
        tool_calls = message.get("tool_calls")
        if message.get("role") != "assistant" or not isinstance(tool_calls, list):
            index += 1
            continue
        call_ids = {
            str(call.get("id", ""))
            for call in tool_calls
            if isinstance(call, dict) and call.get("id")
        }
        if not call_ids:
            index += 1
            continue
        end = index
        cursor = index + 1
        while cursor < len(messages):
            candidate = messages[cursor]
            if candidate.get("role") != "tool":
                break
            if str(candidate.get("tool_call_id", "")) not in call_ids:
                break
            end = cursor
            cursor += 1
        if end > index:
            spans.append((index, end))
            index = end + 1
        else:
            index += 1
    return spans


def _call_identity(call: dict[str, object]) -> tuple[str, dict[str, object]]:
    function = call.get("function")
    if not isinstance(function, dict):
        return "unknown", {}
    name = function.get("name")
    raw_arguments = function.get("arguments")
    arguments: dict[str, object] = {}
    if isinstance(raw_arguments, str):
        try:
            parsed = json.loads(raw_arguments)
        except json.JSONDecodeError:
            parsed = {}
        if isinstance(parsed, dict):
            arguments = parsed
    return name if isinstance(name, str) else "unknown", arguments


def _summarize_tool_result(
    name: str,
    arguments: dict[str, object],
    content: str,
) -> str:
    if name == "run_command":
        return _summarize_command(content)
    if name == "read_file":
        path = arguments.get("path", "unknown")
        return f"{path}; {len(content.splitlines())} lines, {len(content)} chars; {_sample_lines(content, 8)}"
    if name == "search_text":
        return _sample_lines(content, 10)
    if name == "list_files":
        return _sample_lines(content, 30)
    return _compact_inline(content, 700)


def _summarize_command(content: str) -> str:
    lines = content.splitlines()
    important = [
        line
        for line in lines
        if line.startswith("Exit code:")
        or "FAILED" in line
        or "ERROR" in line
        or "Traceback" in line
        or "AssertionError" in line
        or line.startswith("Ran ")
        or line == "OK"
    ]
    if important:
        return _compact_inline(" | ".join(important[:16]), 900)
    return _sample_lines(content, 12)


def _checked_summary(name: str, arguments: dict[str, object]) -> str | None:
    if name == "read_file":
        path = arguments.get("path", "unknown")
        start = arguments.get("start_line")
        end = arguments.get("end_line")
        if start or end:
            return f"read {path} lines {start or 1}-{end or 'end'}"
        return f"read {path}"
    if name == "search_text":
        query = arguments.get("query", "")
        path = arguments.get("path", ".")
        return f"searched {path} for {_compact_inline(str(query), 120)}"
    if name == "list_files":
        return f"listed {arguments.get('path', '.')}"
    if name == ContextPackExtension.READ_OUTPUT_TOOL:
        return f"retrieved offloaded output {arguments.get('output_id', 'unknown')}"
    return None


def _modified_summary(name: str, arguments: dict[str, object]) -> str | None:
    if name == "write_file":
        return f"wrote {arguments.get('path', 'unknown')}"
    if name == "apply_patch":
        return f"patched {arguments.get('path', 'unknown')}"
    return None


def _command_summary(name: str, arguments: dict[str, object]) -> str | None:
    if name != "run_command":
        return None
    argv = arguments.get("argv", [])
    if isinstance(argv, list):
        return _compact_inline(" ".join(str(part) for part in argv), 220)
    return _compact_inline(str(argv), 220)


def _looks_like_failure(content: str) -> bool:
    failure_markers = (
        "Exit code: 1",
        "Exit code: 2",
        "FAILED",
        "ERROR:",
        "Traceback",
        "AssertionError",
    )
    return any(marker in content for marker in failure_markers)


def _bullet_lines(items: Sequence[str]) -> list[str]:
    unique_items = _unique(items)
    if not unique_items:
        return ["- none"]
    return [f"- {_compact_inline(item, 700)}" for item in unique_items]


def _unique(items: Sequence[str]) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for item in items:
        normalized = item.strip()
        if not normalized or normalized in seen:
            continue
        seen.add(normalized)
        result.append(normalized)
    return result


def _positive_int(value: object, default: int) -> int:
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        return default
    return parsed if parsed > 0 else default


def _safe_slug(value: str) -> str:
    slug = "".join(
        character.lower() if character.isalnum() else "-"
        for character in value
    )
    slug = "-".join(part for part in slug.split("-") if part)
    return slug[:40] or "tool"


def _sample_lines(content: str, limit: int) -> str:
    lines = [line.strip() for line in content.splitlines() if line.strip()]
    if len(lines) <= limit:
        return _compact_inline(" | ".join(lines), 900)
    head = lines[: max(1, limit // 2)]
    tail = lines[-max(1, limit // 2) :]
    return _compact_inline(" | ".join([*head, "...", *tail]), 900)


def _compact_inline(text: str, limit: int) -> str:
    return _limit_text(" ".join(text.split()), limit)


def _limit_text(text: str, limit: int) -> str:
    if len(text) <= limit:
        return text
    return text[: max(0, limit - 3)].rstrip() + "..."
