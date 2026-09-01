"""Lifecycle extension hooks for the local coding agent."""

from __future__ import annotations

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
            messages, tools = extension.before_llm_call(
                context,
                step=step,
                messages=messages,
                tools=tools,
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


def _tool_name(definition: ToolDefinition) -> str:
    function = definition.get("function")
    if not isinstance(function, dict):
        return ""
    name = function.get("name")
    return name if isinstance(name, str) else ""
