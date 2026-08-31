"""The core model-tool-model loop."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Callable, Protocol

from coding_agent.checkpoint import CheckpointError, CheckpointStore
from coding_agent.context import ContextManager, ContextStats
from coding_agent.project_memory import ProjectMemoryStore
from coding_agent.skills import SkillRegistry
from coding_agent.tools.registry import ToolRegistry
from coding_agent.trace import NullTrace, Trace
from coding_agent.workspace import Workspace


SYSTEM_PROMPT = """You are a coding agent working in one local project workspace.
Inspect relevant files before editing. Prefer apply_patch for existing files.
Do not read a file back immediately after a successful write or patch unless
the exact content is genuinely needed; rely on tool results and tests instead.
Batch independent tool calls in one response when their contents are known.
For implementation tasks, create or update necessary automated tests and run
the complete relevant test suite after making changes. The final report must
include the exact usage or launch instructions and the real test result.
For a new Python project, prefer standard-library unittest unless the workspace
already uses an available test runner. Do not install a package only to run tests.
For desktop GUI or web UI tasks, include a launch/import smoke check and cover
at least one real UI boundary or callback path when it can be tested without
manual interaction. If a GUI cannot be opened in the environment, say that
explicitly and verify the nearest automated seam instead.
When an execution plan is supplied in task state, follow it before making
changes, complete every planned deliverable before final verification, and use
its acceptance checks to decide whether the task is complete.
For read-only tasks, do not modify files merely to add tests. If meaningful
automated testing is not possible, explain the validation performed instead.
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
PLAN_TASK_DEFINITION = {
    "type": "function",
    "function": {
        "name": "plan_task",
        "description": (
            "Create a concise implementation plan and acceptance checklist "
            "before building a new project from scratch."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "steps": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Two to six concrete implementation steps.",
                },
                "acceptance": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Observable conditions required for completion.",
                },
                "test_strategy": {
                    "type": "string",
                    "description": "How the implementation will be tested.",
                },
            },
            "required": ["steps", "acceptance", "test_strategy"],
            "additionalProperties": False,
        },
    },
}
APPROVE_PLAN_DEFINITION = {
    "type": "function",
    "function": {
        "name": "approve_plan",
        "description": "Approve a plan only when it preserves every explicit requirement.",
        "parameters": {
            "type": "object",
            "properties": {},
            "additionalProperties": False,
        },
    },
}
REJECT_PLAN_DEFINITION = {
    "type": "function",
    "function": {
        "name": "reject_plan",
        "description": "Reject a plan that omits or substitutes an explicit requirement.",
        "parameters": {
            "type": "object",
            "properties": {
                "reason": {
                    "type": "string",
                    "description": "Concise instructions for correcting the plan.",
                }
            },
            "required": ["reason"],
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
PLANNING_GATE_PROMPT = """Plan this from-scratch software project before editing.
Call ask_user only if an essential choice is missing and cannot be safely
inferred. Otherwise call plan_task with a concise implementation plan,
observable acceptance checks, and an automated test strategy. Include the
launchable entry point and usage documentation in the plan. For a nontrivial
application, keep the launch entry point thin and separate from testable core
logic and the user-interface or external-I/O boundary. Do not call local tools
or provide a plain-text answer yet. For a new Python project, use the
standard-library unittest runner unless the workspace already provides another
test runner; never plan to install a test dependency.
For desktop GUI or web UI projects, include a launch/import smoke check and a
test or harness that exercises at least one UI callback or request path.
"""
PLAN_REVIEW_PROMPT = """Review a proposed implementation plan against the
original user request. Call approve_plan only if every explicit requirement is
covered without substitution. Otherwise call reject_plan with a concise reason.
Do not replace a requested interface (desktop GUI, web, CLI, or API), platform,
feature, or user-adjustable setting with an easier alternative. Keep automated
tests, usage documentation, and a thin launch entry point separate from the
testable core and interface boundary in the corrected plan.
Call exactly one control action and do not provide a plain-text answer.
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
    usage: "TokenUsage | None" = None


@dataclass(frozen=True)
class TokenUsage:
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    cache_hit_tokens: int = 0


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
    creation_task: bool = False
    modified_files: list[str] = field(default_factory=list)
    latest_command: str = "not run"
    last_error: str = "none"
    changes_pending_verification: bool = False
    implementation_changes_pending: bool = False
    full_tests_passed: bool = False
    plan: list[str] = field(default_factory=list)
    acceptance: list[str] = field(default_factory=list)
    test_strategy: str = "not planned"

    @property
    def ready_to_finalize(self) -> bool:
        return (
            self.full_tests_passed
            and not self.changes_pending_verification
            and not self.missing_deliverables
        )

    @property
    def missing_deliverables(self) -> list[str]:
        if not self.creation_task:
            return []
        paths = [path.replace("\\", "/").casefold() for path in self.modified_files]
        missing: list[str] = []
        implementation = [
            path
            for path in paths
            if path.endswith(".py") and not _is_test_path(path)
        ]
        if not implementation:
            missing.append("implementation source")
        if not any(_is_entry_point(path) for path in implementation):
            missing.append("entry point")
        if not any(_is_test_path(path) for path in paths):
            missing.append("automated tests")
        if not any(path.rsplit("/", 1)[-1].startswith("readme") for path in paths):
            missing.append("usage documentation")
        return missing

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
            if str(path).lower().endswith(".py"):
                self.implementation_changes_pending = True
            self.full_tests_passed = False
            self.phase = "implement" if self.missing_deliverables else "verify"
            self.last_error = "none"
        elif call.name == "run_command":
            if "Exit code: 0" in result:
                self.latest_command = "passed (exit 0)"
                if self.changes_pending_verification and _is_full_test_command(call):
                    if _test_command_ran_tests(call, result):
                        self.changes_pending_verification = False
                        self.implementation_changes_pending = False
                        self.full_tests_passed = True
                    else:
                        self.latest_command = "not verified (no tests executed)"
                        self.phase = "verify"
                        self.last_error = (
                            "Test command exited 0 but no tests were executed"
                        )
                        return
                if self.ready_to_finalize:
                    self.phase = "review" if self.plan else "finalize"
                elif self.missing_deliverables:
                    self.phase = "implement"
                elif self.changes_pending_verification:
                    self.phase = "verify"
                else:
                    self.phase = "inspect"
                self.last_error = "none"
            else:
                self.latest_command = "failed"
                self.phase = "finalize" if self.ready_to_finalize else "repair"
                self.last_error = result[:200]

    def contract_message(self) -> dict[str, object]:
        plan = " | ".join(self.plan) or "none"
        acceptance = " | ".join(self.acceptance) or "none"
        return {
            "role": "system",
            "content": (
                "<task_contract>\n"
                f"Goal: {self.goal}\n"
                f"Execution plan: {plan}\n"
                f"Acceptance checks: {acceptance}\n"
                f"Test strategy: {self.test_strategy}\n"
                "</task_contract>"
            ),
        }

    def message(self, remaining_rounds: int) -> dict[str, object]:
        files = ", ".join(self.modified_files) or "none"
        missing = ", ".join(self.missing_deliverables) or "none"
        next_focus = {
            "inspect": "inspect only the files needed for the task",
            "implement": (
                "create the missing deliverables before verification; do not reread "
                "files merely to confirm successful writes"
            ),
            "verify": "run focused verification for the changes",
            "repair": (
                "diagnose the latest failure before changing more code: reproduce "
                "the symptom, compare it with the user's report, test one focused "
                "hypothesis, apply the smallest fix, then rerun the failing path "
                "and relevant tests"
            ),
            "review": "review every plan item and finish any missing deliverable",
            "finalize": (
                "Return a final answer now. Do not call another tool unless an "
                "explicit acceptance requirement remains unverified"
            ),
        }[self.phase]
        return {
            "role": "system",
            "content": (
                "<task_state>\n"
                f"Phase: {self.phase}\n"
                f"Modified files: {files}\n"
                f"Missing deliverables: {missing}\n"
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
            "creation_task": self.creation_task,
            "modified_files": self.modified_files,
            "latest_command": self.latest_command,
            "last_error": self.last_error,
            "changes_pending_verification": self.changes_pending_verification,
            "implementation_changes_pending": self.implementation_changes_pending,
            "full_tests_passed": self.full_tests_passed,
            "plan": self.plan,
            "acceptance": self.acceptance,
            "test_strategy": self.test_strategy,
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
        plan = data.get("plan", [])
        acceptance = data.get("acceptance", [])
        if not isinstance(plan, list) or not all(isinstance(item, str) for item in plan):
            raise CheckpointError("Invalid checkpoint plan")
        if not isinstance(acceptance, list) or not all(
            isinstance(item, str) for item in acceptance
        ):
            raise CheckpointError("Invalid checkpoint acceptance checks")
        return cls(
            goal=str(data.get("goal", "")),
            phase=str(data.get("phase", "inspect")),
            creation_task=bool(data.get("creation_task", False)),
            modified_files=modified_files,
            latest_command=str(data.get("latest_command", "not run")),
            last_error=str(data.get("last_error", "none")),
            changes_pending_verification=bool(
                data.get("changes_pending_verification", False)
            ),
            implementation_changes_pending=bool(
                data.get("implementation_changes_pending", False)
            ),
            full_tests_passed=bool(data.get("full_tests_passed", False)),
            plan=plan,
            acceptance=acceptance,
            test_strategy=str(data.get("test_strategy", "not planned")),
        )


@dataclass(frozen=True)
class SessionTaskSummary:
    task: str
    answer: str
    modified_files: tuple[str, ...]
    latest_command: str


def _is_test_path(path: str) -> bool:
    normalized = path.replace("\\", "/").casefold()
    name = normalized.rsplit("/", 1)[-1]
    return (
        normalized.startswith("tests/")
        or "/tests/" in normalized
        or name.startswith("test_")
        or name.endswith("_test.py")
    )


def _is_entry_point(path: str) -> bool:
    name = path.replace("\\", "/").casefold().rsplit("/", 1)[-1]
    return name in {"main.py", "__main__.py", "app.py"}


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
    return len(arguments) >= 3 and arguments[1:3] == ["-m", "unittest"]


def _test_command_ran_tests(call: ToolCall, result: str) -> bool:
    argv = call.arguments.get("argv")
    if not isinstance(argv, list) or not argv:
        return False
    program = str(argv[0]).replace("\\", "/").rsplit("/", 1)[-1].lower()
    if program in {"pytest", "pytest.exe", "py.test", "py.test.exe"}:
        match = re.search(r"\b(\d+)\s+passed\b", result)
    else:
        match = re.search(r"\bRan\s+(\d+)\s+tests?\b", result)
    return match is not None and int(match.group(1)) > 0


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


def _needs_creation_plan(task: str) -> bool:
    normalized = " ".join(task.lower().split())
    return (
        ("从零" in normalized and any(word in normalized for word in ("创建", "搭建", "开发", "实现")))
        or "from scratch" in normalized
    )


def _compact_text(text: str, limit: int) -> str:
    compacted = " ".join(text.split())
    if len(compacted) <= limit:
        return compacted
    return compacted[: limit - 3].rstrip() + "..."


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
        context_manager: ContextManager | None = None,
        memory_store: ProjectMemoryStore | None = None,
        skill_registry: SkillRegistry | None = None,
    ) -> None:
        self.model = model
        self.workspace = workspace
        self.context_manager = context_manager or ContextManager(enabled=False)
        self.tools = ToolRegistry(
            workspace,
            cache_reads=self.context_manager.enabled,
        )
        self.max_steps = max_steps
        self.on_event = on_event or (lambda _message: None)
        self.trace = trace or NullTrace()
        self.checkpoint_store = checkpoint_store
        self.memory_store = memory_store
        self.skill_registry = skill_registry or SkillRegistry.load_builtin()
        self._pending_task: str | None = None
        self._prompt_tokens = 0
        self._completion_tokens = 0
        self._cache_hit_tokens = 0
        self._session_tasks: list[SessionTaskSummary] = []

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
        self.context_manager.start_task(reset=not self._session_tasks)
        self.tools.start_task()
        self.on_event("[状态] 正在分析任务")
        self.trace.record("task_start", task=task)
        messages: list[dict[str, object]] = [
            {"role": "system", "content": SYSTEM_PROMPT},
        ]
        project_memory = self._project_memory_message()
        if project_memory is not None:
            messages.append(project_memory)
            self.trace.record("project_memory_attached")
        session_context = self._session_context_message()
        if session_context is not None:
            messages.append(session_context)
            self.trace.record(
                "session_context_attached",
                completed_tasks=len(self._session_tasks),
            )
        selected_skills = self.skill_registry.select(task)
        if selected_skills:
            skill_names = [skill.name for skill in selected_skills]
            messages.extend(skill.message() for skill in selected_skills)
            self.trace.record("skills_selected", skills=skill_names)
            self.on_event(f"[技能] 已启用：{', '.join(skill_names)}")
        messages.append({"role": "user", "content": task})
        state = TaskState(task)

        if _needs_clarification_gate(task):
            question = self._run_clarification_gate(task)
            if question is not None:
                return question

        if _needs_creation_plan(task):
            question = self._run_creation_planning_gate(task, state)
            if question is not None:
                return question

        answer = self._continue(task, messages, state, step=1)
        self._remember_completed_task(task, answer, state)
        return answer

    def _project_memory_message(self) -> dict[str, object] | None:
        if self.memory_store is None:
            return None
        return self.memory_store.build_message(self.workspace)

    def _session_context_message(self) -> dict[str, object] | None:
        if not self._session_tasks:
            return None
        recent = self._session_tasks[-5:]
        lines = [
            "<session_context>",
            "The user is continuing in the same CLI session and workspace. "
            "Treat the new request as a follow-up to the current project unless "
            "the user clearly asks to switch projects.",
            "Use this as orientation only: inspect the current workspace files "
            "before editing or explaining behavior.",
            "Recent completed tasks:",
        ]
        for index, summary in enumerate(recent, start=1):
            files = ", ".join(summary.modified_files) or "none"
            lines.extend(
                [
                    f"{index}. Task: {_compact_text(summary.task, 240)}",
                    f"   Result: {_compact_text(summary.answer, 320)}",
                    f"   Modified files: {files}",
                    f"   Latest command: {summary.latest_command}",
                ]
            )
        lines.append("</session_context>")
        return {"role": "system", "content": "\n".join(lines)}

    def _remember_completed_task(
        self,
        task: str,
        answer: str,
        state: TaskState,
    ) -> None:
        self._session_tasks.append(
            SessionTaskSummary(
                task=task,
                answer=answer,
                modified_files=tuple(state.modified_files),
                latest_command=state.latest_command,
            )
        )
        del self._session_tasks[:-5]
        self.trace.record(
            "session_context_updated",
            completed_tasks=len(self._session_tasks),
            modified_files=state.modified_files,
        )
        if self.memory_store is not None:
            self.memory_store.update_after_task(
                self.workspace,
                task=task,
                answer=answer,
                modified_files=state.modified_files,
                latest_command=state.latest_command,
            )
            self.trace.record("project_memory_updated")

    def _run_creation_planning_gate(
        self,
        task: str,
        state: TaskState,
    ) -> str | None:
        self.on_event("[状态] 正在制定实施计划")
        messages: list[dict[str, object]] = [
            {"role": "system", "content": PLANNING_GATE_PROMPT},
            {"role": "user", "content": task},
        ]
        self.trace.record("model_request", step=0, planning_gate=True)
        reply = self._complete_model_request(
            messages,
            [PLAN_TASK_DEFINITION, ASK_USER_DEFINITION],
            step=0,
            error_prefix="Planning",
        )
        self.trace.record(
            "model_reply",
            step=0,
            content=reply.content,
            tools=[call.name for call in reply.tool_calls],
            planning_gate=True,
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

        plan_call = next(
            (call for call in reply.tool_calls if call.name == "plan_task"),
            None,
        )
        if plan_call is None or len(reply.tool_calls) != 1:
            raise AgentError("Model did not provide a valid implementation plan")
        steps, acceptance, test_strategy = self._parse_plan(plan_call)
        steps, acceptance, test_strategy = self._review_creation_plan(
            task,
            steps,
            acceptance,
            test_strategy,
        )

        state.plan = steps
        state.acceptance = acceptance
        state.test_strategy = test_strategy
        state.creation_task = True
        self.on_event(f"[计划] {' → '.join(steps)}")
        self.trace.record(
            "task_planned",
            step=0,
            steps=steps,
            acceptance=acceptance,
            test_strategy=test_strategy,
        )
        return None

    def _review_creation_plan(
        self,
        task: str,
        steps: list[str],
        acceptance: list[str],
        test_strategy: str,
    ) -> tuple[list[str], list[str], str]:
        self.on_event("[状态] 正在校验实施计划")
        proposed_plan = json.dumps(
            {
                "steps": steps,
                "acceptance": acceptance,
                "test_strategy": test_strategy,
            },
            ensure_ascii=False,
        )
        messages: list[dict[str, object]] = [
            {"role": "system", "content": PLAN_REVIEW_PROMPT},
            {
                "role": "user",
                "content": (
                    f"Original request:\n{task}\n\n"
                    f"Proposed plan:\n{proposed_plan}"
                ),
            },
        ]
        self.trace.record("model_request", step=0, plan_review=True)
        reply = self._complete_model_request(
            messages,
            [APPROVE_PLAN_DEFINITION, REJECT_PLAN_DEFINITION],
            step=0,
            error_prefix="Plan review",
        )
        self.trace.record(
            "model_reply",
            step=0,
            content=reply.content,
            tools=[call.name for call in reply.tool_calls],
            plan_review=True,
        )
        if len(reply.tool_calls) != 1:
            raise AgentError("Model did not provide a valid plan review")
        decision = reply.tool_calls[0]
        if decision.name == "approve_plan":
            self.trace.record("plan_approved", step=0)
            return steps, acceptance, test_strategy
        if decision.name == "reject_plan":
            reason = str(decision.arguments.get("reason", "")).strip()
            if not reason:
                raise AgentError("Plan review rejected without a reason")
            return self._revise_creation_plan(task, proposed_plan, reason)
        raise AgentError("Model did not provide a valid plan review")

    def _revise_creation_plan(
        self,
        task: str,
        proposed_plan: str,
        reason: str,
    ) -> tuple[list[str], list[str], str]:
        self.on_event("[状态] 正在修订实施计划")
        messages: list[dict[str, object]] = [
            {"role": "system", "content": PLANNING_GATE_PROMPT},
            {
                "role": "user",
                "content": (
                    f"Original request:\n{task}\n\n"
                    f"Rejected plan:\n{proposed_plan}\n\n"
                    f"Review feedback:\n{reason}\n\n"
                    "Return a corrected plan that addresses the feedback."
                ),
            },
        ]
        self.trace.record("model_request", step=0, plan_revision=True)
        reply = self._complete_model_request(
            messages,
            [PLAN_TASK_DEFINITION],
            step=0,
            error_prefix="Plan revision",
        )
        self.trace.record(
            "model_reply",
            step=0,
            content=reply.content,
            tools=[call.name for call in reply.tool_calls],
            plan_revision=True,
        )
        if len(reply.tool_calls) != 1 or reply.tool_calls[0].name != "plan_task":
            raise AgentError("Model did not provide a corrected implementation plan")
        self.trace.record("plan_revised", step=0, reason=reason)
        return self._parse_plan(reply.tool_calls[0])

    def _parse_plan(self, call: ToolCall) -> tuple[list[str], list[str], str]:
        steps = self._string_list(call.arguments.get("steps"), "plan steps")
        acceptance = self._string_list(
            call.arguments.get("acceptance"),
            "acceptance checks",
        )
        test_strategy = str(call.arguments.get("test_strategy", "")).strip()
        if not 2 <= len(steps) <= 6 or not acceptance or not test_strategy:
            raise AgentError("Model provided an incomplete implementation plan")
        return steps, acceptance, test_strategy

    @staticmethod
    def _string_list(value: object, label: str) -> list[str]:
        if not isinstance(value, list):
            raise AgentError(f"Invalid {label}")
        items = [str(item).strip() for item in value]
        if not all(items):
            raise AgentError(f"Invalid {label}")
        return items

    def _run_clarification_gate(self, task: str) -> str | None:
        self.on_event("[状态] 正在确认需求")
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
            self._observe_error_usage(error, step=0)
            self.trace.record(
                "task_error",
                step=0,
                error_type=type(error).__name__,
            )
            raise AgentError(f"Clarification check failed: {error}") from error
        self._observe_usage(reply, step=0)
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
        self.on_event("[状态] 正在恢复任务")
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
        answer = self._continue(
            task,
            messages,
            state,
            step=step,
            pending_calls=pending_calls,
            next_call_index=next_call_index,
        )
        self._remember_completed_task(task, answer, state)
        return answer

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
            if state.ready_to_finalize and not state.plan:
                return self._finalize_after_verification(messages, state, step + 1)
            step += 1

        while step <= self.max_steps:
            if step > 1:
                self.on_event("[状态] 正在规划下一步")
            self.trace.record("model_request", step=step)
            action_tools = self._action_tools(state)
            request_messages = self._build_request_context(
                messages,
                [state.message(self.max_steps - step + 1)],
                step=step,
                contract_messages=[state.contract_message()],
                tools=action_tools,
            )
            reply = self._complete_action_request(
                request_messages,
                step,
                action_tools,
            )

            self.trace.record(
                "model_reply",
                step=step,
                content=reply.content,
                tools=[call.name for call in reply.tool_calls],
            )
            self.context_manager.record_step(
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
                    if state.missing_deliverables:
                        self.trace.record(
                            "completion_deferred",
                            step=step,
                            reason="missing_deliverables",
                            missing=state.missing_deliverables,
                        )
                        messages.extend(
                            [
                                {"role": "assistant", "content": reply.content},
                                {
                                    "role": "system",
                                    "content": (
                                        "The task is not complete. Create these missing "
                                        "deliverables before answering: "
                                        + ", ".join(state.missing_deliverables)
                                        + "."
                                    ),
                                },
                            ]
                        )
                        step += 1
                        continue
                    if state.implementation_changes_pending:
                        self.trace.record(
                            "completion_deferred",
                            step=step,
                            reason="implementation_not_verified",
                        )
                        messages.extend(
                            [
                                {"role": "assistant", "content": reply.content},
                                {
                                    "role": "system",
                                    "content": (
                                        "Python code has changed but the complete "
                                        "relevant test suite has not passed since "
                                        "the latest change. Continue with tools to "
                                        "add or update tests as needed and run the "
                                        "full suite before reporting completion."
                                    ),
                                },
                            ]
                        )
                        step += 1
                        continue
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

            if state.ready_to_finalize and not state.plan:
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

    def _complete_action_request(
        self,
        request_messages: list[dict[str, object]],
        step: int,
        tools: list[dict[str, object]],
    ) -> ModelReply:
        return self._complete_model_request(
            request_messages,
            tools,
            step=step,
            error_prefix="Model request",
        )

    def _action_tools(self, state: TaskState) -> list[dict[str, object]]:
        definitions = self.tools.definitions
        if state.phase == "verify" and state.changes_pending_verification:
            allowed = {"write_file", "apply_patch", "run_command"}
            definitions = [
                definition
                for definition in definitions
                if definition.get("function", {}).get("name") in allowed
            ]
        return [*definitions, ASK_USER_DEFINITION]

    def _build_request_context(
        self,
        messages: list[dict[str, object]],
        tail_messages: list[dict[str, object]],
        *,
        step: int,
        contract_messages: list[dict[str, object]] | None = None,
        tools: list[dict[str, object]] | None = None,
    ) -> list[dict[str, object]]:
        request = self.context_manager.build(
            messages,
            contract_messages=contract_messages,
            state_messages=tail_messages,
            tools=tools,
        )
        stats: ContextStats = self.context_manager.last_stats
        self.trace.record(
            "context_built",
            step=step,
            original_characters=stats.original_characters,
            sent_characters=stats.sent_characters,
            saved_characters=stats.saved_characters,
            original_tokens=stats.original_tokens,
            sent_tokens=stats.sent_tokens,
            saved_tokens=stats.saved_tokens,
            tool_definition_tokens=stats.tool_definition_tokens,
            summarized_messages=stats.summarized_messages,
            retrieved_entries=stats.retrieved_entries,
            context_mode=self.context_manager.mode,
        )
        if stats.summarized_messages:
            percent = round(
                stats.saved_tokens * 100 / max(1, stats.original_tokens)
            )
            self.on_event(
                f"[上下文] 已压缩 {stats.summarized_messages} 条旧消息，"
                f"本轮预计 Token 减少约 {percent}%"
            )
        return request

    def _observe_usage(self, reply: ModelReply, *, step: int) -> None:
        usage = reply.usage
        if usage is None:
            return
        self._prompt_tokens += usage.prompt_tokens
        self._completion_tokens += usage.completion_tokens
        self._cache_hit_tokens += usage.cache_hit_tokens
        self.context_manager.observe_prompt_usage(usage.prompt_tokens)
        total = self._prompt_tokens + self._completion_tokens
        self.on_event(
            f"[Token] 本次输入 {usage.prompt_tokens}，输出 {usage.completion_tokens}；"
            f"本次会话累计 {total}"
        )
        self.trace.record(
            "token_usage",
            step=step,
            prompt_tokens=usage.prompt_tokens,
            completion_tokens=usage.completion_tokens,
            total_tokens=usage.total_tokens,
            cache_hit_tokens=usage.cache_hit_tokens,
            cumulative_prompt_tokens=self._prompt_tokens,
            cumulative_completion_tokens=self._completion_tokens,
            cumulative_cache_hit_tokens=self._cache_hit_tokens,
        )

    def _observe_error_usage(self, error: Exception, *, step: int) -> None:
        usage = getattr(error, "usage", None)
        if isinstance(usage, TokenUsage):
            self._observe_usage(ModelReply(content=None, usage=usage), step=step)

    def _complete_model_request(
        self,
        request_messages: list[dict[str, object]],
        tools: list[dict[str, object]],
        *,
        step: int,
        error_prefix: str,
    ) -> ModelReply:
        retry_messages = request_messages
        for attempt in range(2):
            try:
                reply = self.model.complete(retry_messages, tools)
                self._observe_usage(reply, step=step)
                return reply
            except Exception as error:
                self._observe_error_usage(error, step=step)
                if attempt == 0:
                    self.on_event("[状态] 模型响应异常，正在重试")
                    self.trace.record(
                        "model_retry",
                        step=step,
                        error_type=type(error).__name__,
                        error=str(error),
                    )
                    retry_messages = [
                        *request_messages,
                        {
                            "role": "system",
                            "content": (
                                "The previous model response could not be processed: "
                                f"{error}. Retry this same step. If calling a tool, "
                                "return arguments that exactly match its JSON schema."
                            ),
                        },
                    ]
                    continue
                self.trace.record(
                    "task_error",
                    step=step,
                    error_type=type(error).__name__,
                )
                raise AgentError(f"{error_prefix} failed after retry: {error}") from error
        raise AssertionError("unreachable")

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
            if call.name in {"list_files", "read_file", "search_text"}:
                self.on_event("[状态] 正在检查项目")
            elif call.name in {"write_file", "apply_patch"}:
                path = str(call.arguments.get("path", "项目文件"))
                self.on_event(f"[状态] 正在修改 {path}")
            elif _is_full_test_command(call):
                self.on_event("[状态] 正在运行测试")
            elif call.name == "run_command":
                self.on_event("[状态] 正在执行验证命令")
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
            if _is_full_test_command(call):
                if "Exit code: 0" in result:
                    self.on_event("[状态] 测试通过，正在整理结果")
                else:
                    self.on_event("[状态] 测试失败，正在分析原因")
            state.update(call, result, success)
            self.context_manager.record_tool(
                step=step,
                name=call.name,
                arguments=call.arguments,
                result=result,
                success=success,
                version=str(self.tools.last_metadata.get("version", "")),
                file_content=str(self.tools.last_metadata.get("content", "")),
            )
            self.trace.record(
                "tool_result",
                step=step,
                tool=call.name,
                success=success,
                result=result,
            )
            if call.name == "read_file" and result.startswith(
                "unchanged read cache hit:"
            ):
                self.trace.record(
                    "read_cache_hit",
                    step=step,
                    path=call.arguments.get("path", ""),
                )
            distilled_result = self.context_manager.distill_tool_result(
                call.name,
                call.arguments,
                result,
                success=success,
            )
            messages.append(
                {
                    "role": "tool",
                    "tool_call_id": call.id,
                    "content": distilled_result,
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
                "report in Chinese based on the completed work and tests. "
                "Include exact usage or launch instructions for the result."
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
        self.on_event("[状态] 需要补充信息")
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
        self.on_event("[状态] 正在整理结果")
        request_messages = self._build_request_context(
            messages,
            [
                state.message(0),
                {"role": "system", "content": instruction},
            ],
            step=step,
            contract_messages=[state.contract_message()],
            tools=[],
        )
        self.trace.record(
            "model_request",
            step=step,
            finalization=True,
            reason=reason,
        )
        try:
            reply = self.model.complete(request_messages, [])
        except Exception as error:
            self._observe_error_usage(error, step=step)
            self.trace.record(
                "task_error",
                step=step,
                error_type=type(error).__name__,
            )
            raise AgentError(f"Final model request failed: {error}") from error
        self._observe_usage(reply, step=step)

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
            self._observe_error_usage(error, step=step)
            self.trace.record(
                "task_error",
                step=step,
                error_type=type(error).__name__,
            )
            raise AgentError(f"Final model retry failed: {error}") from error
        self._observe_usage(reply, step=step)

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
