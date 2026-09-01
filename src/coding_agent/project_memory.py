"""Persistent per-project memory across agent conversations."""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

from coding_agent.workspace import PROTECTED_DIRECTORIES, PROTECTED_FILES, Workspace


MAX_STRUCTURE_ITEMS = 300
MAX_DECISIONS = 40
MAX_COMMANDS = 20
MAX_TASKS = 50


@dataclass
class ProjectTaskSummary:
    task: str
    answer: str
    modified_files: list[str] = field(default_factory=list)
    latest_command: str = "not run"
    completed_at: str = ""

    def to_data(self) -> dict[str, object]:
        return {
            "task": self.task,
            "answer": self.answer,
            "modified_files": self.modified_files,
            "latest_command": self.latest_command,
            "completed_at": self.completed_at,
        }

    @classmethod
    def from_data(cls, data: object) -> "ProjectTaskSummary | None":
        if not isinstance(data, dict):
            return None
        modified_files = data.get("modified_files", [])
        if not isinstance(modified_files, list):
            modified_files = []
        return cls(
            task=str(data.get("task", "")),
            answer=str(data.get("answer", "")),
            modified_files=[str(item) for item in modified_files],
            latest_command=str(data.get("latest_command", "not run")),
            completed_at=str(data.get("completed_at", "")),
        )


@dataclass
class ProjectMemory:
    workspace: str
    project_id: str
    display_name: str = ""
    sort_order: int = 0
    structure: list[str] = field(default_factory=list)
    user_decisions: list[str] = field(default_factory=list)
    launch_commands: list[str] = field(default_factory=list)
    tasks: list[ProjectTaskSummary] = field(default_factory=list)
    updated_at: str = ""

    def to_data(self) -> dict[str, object]:
        return {
            "version": 1,
            "workspace": self.workspace,
            "project_id": self.project_id,
            "display_name": self.display_name,
            "sort_order": self.sort_order,
            "structure": self.structure,
            "user_decisions": self.user_decisions,
            "launch_commands": self.launch_commands,
            "tasks": [task.to_data() for task in self.tasks],
            "updated_at": self.updated_at,
        }

    @classmethod
    def from_data(cls, data: object, *, workspace: str, project_id: str) -> "ProjectMemory":
        if not isinstance(data, dict) or data.get("version") != 1:
            return cls(workspace=workspace, project_id=project_id)
        tasks = [
            task
            for item in data.get("tasks", [])
            if (task := ProjectTaskSummary.from_data(item)) is not None
        ]
        return cls(
            workspace=str(data.get("workspace", workspace)),
            project_id=str(data.get("project_id", project_id)),
            display_name=str(data.get("display_name", "")),
            sort_order=_int_value(data.get("sort_order", 0)),
            structure=_string_list(data.get("structure")),
            user_decisions=_string_list(data.get("user_decisions")),
            launch_commands=_string_list(data.get("launch_commands")),
            tasks=tasks,
            updated_at=str(data.get("updated_at", "")),
        )


class ProjectMemoryStore:
    """JSON-backed project memory keyed by workspace path."""

    def __init__(self, directory: Path) -> None:
        self.directory = directory
        self.directory.mkdir(parents=True, exist_ok=True)

    def project_id(self, workspace: Workspace) -> str:
        digest = hashlib.sha256(str(workspace.root).encode("utf-8")).hexdigest()
        return digest[:16]

    def path_for(self, workspace: Workspace) -> Path:
        return self.directory / f"{self.project_id(workspace)}.json"

    def load(self, workspace: Workspace) -> ProjectMemory:
        project_id = self.project_id(workspace)
        path = self.path_for(workspace)
        if not path.exists():
            return ProjectMemory(workspace=str(workspace.root), project_id=project_id)
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return ProjectMemory(workspace=str(workspace.root), project_id=project_id)
        return ProjectMemory.from_data(
            data,
            workspace=str(workspace.root),
            project_id=project_id,
        )

    def save(self, memory: ProjectMemory) -> None:
        path = self.directory / f"{memory.project_id}.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(".tmp")
        temporary.write_text(
            json.dumps(memory.to_data(), ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        temporary.replace(path)

    def rename_project(self, project_id: str, display_name: str) -> ProjectMemory:
        name = display_name.strip()
        if not name:
            raise ValueError("Project name cannot be empty")
        for memory in self.list_projects():
            if memory.project_id == project_id:
                memory.display_name = name
                memory.updated_at = _now()
                self.save(memory)
                return memory
        raise KeyError(f"Project not found: {project_id}")

    def delete_project(self, project_id: str) -> None:
        path = self.directory / f"{project_id}.json"
        if not path.exists():
            raise KeyError(f"Project not found: {project_id}")
        path.unlink()

    def reorder_projects(self, project_ids: list[str]) -> list[ProjectMemory]:
        projects = self.list_projects()
        by_id = {project.project_id: project for project in projects}
        ordered_ids = [project_id for project_id in project_ids if project_id in by_id]
        ordered_ids.extend(project.project_id for project in projects if project.project_id not in ordered_ids)
        for index, project_id in enumerate(ordered_ids):
            memory = by_id[project_id]
            memory.sort_order = index
            memory.updated_at = _now()
            self.save(memory)
        return self.list_projects()

    def update_after_task(
        self,
        workspace: Workspace,
        *,
        task: str,
        answer: str,
        modified_files: list[str],
        latest_command: str,
    ) -> ProjectMemory:
        memory = self.load(workspace)
        memory.workspace = str(workspace.root)
        if not memory.display_name:
            memory.display_name = workspace.root.name or str(workspace.root)
        memory.structure = snapshot_structure(workspace)
        memory.user_decisions = _merge_limited(
            memory.user_decisions,
            extract_user_decisions(task),
            MAX_DECISIONS,
        )
        memory.launch_commands = _merge_limited(
            memory.launch_commands,
            extract_launch_commands(answer),
            MAX_COMMANDS,
        )
        memory.tasks.append(
            ProjectTaskSummary(
                task=_compact(task, 500),
                answer=_compact(answer, 800),
                modified_files=list(modified_files),
                latest_command=latest_command,
                completed_at=_now(),
            )
        )
        del memory.tasks[:-MAX_TASKS]
        memory.updated_at = _now()
        self.save(memory)
        return memory

    def build_message(self, workspace: Workspace) -> dict[str, object] | None:
        memory = self.load(workspace)
        if not (
            memory.structure
            or memory.user_decisions
            or memory.launch_commands
            or memory.tasks
        ):
            return None
        lines = [
            "<project_memory>",
            f"Workspace: {memory.workspace}",
            "This is persistent orientation from previous conversations. "
            "Use it to understand continuity, but inspect current files before "
            "editing or explaining the project.",
        ]
        if memory.structure:
            lines.append("Project structure:")
            lines.extend(f"- {path}" for path in memory.structure[:80])
        if memory.user_decisions:
            lines.append("User decisions:")
            lines.extend(f"- {decision}" for decision in memory.user_decisions[-12:])
        if memory.launch_commands:
            lines.append("Known launch commands:")
            lines.extend(f"- {command}" for command in memory.launch_commands[-8:])
        if memory.tasks:
            lines.append("Recent task history:")
            for item in memory.tasks[-8:]:
                files = ", ".join(item.modified_files) or "none"
                lines.extend(
                    [
                        f"- Task: {_compact(item.task, 220)}",
                        f"  Result: {_compact(item.answer, 280)}",
                        f"  Modified files: {files}",
                    ]
                )
        lines.append("</project_memory>")
        return {"role": "system", "content": "\n".join(lines)}

    def list_projects(self) -> list[ProjectMemory]:
        projects: list[ProjectMemory] = []
        for path in sorted(self.directory.glob("*.json")):
            try:
                data = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                continue
            workspace = str(data.get("workspace", ""))
            project_id = str(data.get("project_id", path.stem))
            projects.append(
                ProjectMemory.from_data(
                    data,
                    workspace=workspace,
                    project_id=project_id,
                )
            )
        return sorted(projects, key=lambda item: (item.sort_order, item.display_name.casefold(), item.workspace.casefold()))


def snapshot_structure(workspace: Workspace) -> list[str]:
    items: list[str] = []
    for path in sorted(workspace.root.rglob("*")):
        relative = path.relative_to(workspace.root)
        if _is_protected_relative(relative):
            continue
        suffix = "/" if path.is_dir() else ""
        items.append(relative.as_posix() + suffix)
        if len(items) >= MAX_STRUCTURE_ITEMS:
            items.append("...")
            break
    return items


def extract_user_decisions(task: str) -> list[str]:
    decisions: list[str] = []
    marker = "User clarification:"
    if marker in task:
        clarification = task.split(marker, 1)[1].strip()
        if clarification:
            decisions.append(_compact(clarification, 300))
    for line in task.splitlines():
        stripped = line.strip()
        if any(word in stripped for word in ("使用", "保持", "不要", "需要", "改为")):
            decisions.append(_compact(stripped, 300))
    return _dedupe(decisions)


def extract_launch_commands(answer: str) -> list[str]:
    commands: list[str] = []
    for line in answer.splitlines():
        stripped = line.strip().strip("`")
        stripped = re.sub(r"^(?:powershell|cmd|bash)\s*>\s*", "", stripped)
        if re.match(r"^(?:python|py|pytest|coding-agent)(?:\.exe)?\b", stripped):
            commands.append(_compact(stripped, 240))
    return _dedupe(commands)


def _is_protected_relative(path: Path) -> bool:
    if path.name in PROTECTED_FILES:
        return True
    return any(part in PROTECTED_DIRECTORIES for part in path.parts)


def _merge_limited(existing: list[str], new_items: list[str], limit: int) -> list[str]:
    merged = _dedupe([*existing, *new_items])
    return merged[-limit:]


def _dedupe(items: list[str]) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for item in items:
        normalized = item.casefold()
        if not item or normalized in seen:
            continue
        seen.add(normalized)
        result.append(item)
    return result


def _string_list(value: object) -> list[str]:
    if not isinstance(value, list):
        return []
    return [str(item) for item in value]


def _int_value(value: object) -> int:
    try:
        return int(value or 0)
    except (TypeError, ValueError):
        return 0


def _compact(text: str, limit: int) -> str:
    compacted = " ".join(text.split())
    if len(compacted) <= limit:
        return compacted
    return compacted[: limit - 3].rstrip() + "..."


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()
