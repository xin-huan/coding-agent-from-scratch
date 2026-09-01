"""Lifecycle extension hooks for the local coding agent."""

from __future__ import annotations

import json
import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Protocol, Sequence

from coding_agent.project_memory import ProjectMemoryStore
from coding_agent.skills import SkillRegistry
from coding_agent.trace import Trace
from coding_agent.workspace import Workspace


Message = dict[str, object]
ToolDefinition = dict[str, object]


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
