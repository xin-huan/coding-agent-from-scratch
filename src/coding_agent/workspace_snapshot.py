"""Workspace file snapshots for restoring chat session nodes."""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from uuid import uuid4

from coding_agent.workspace import PROTECTED_DIRECTORIES, PROTECTED_FILES, Workspace


MAX_SNAPSHOT_FILE_CHARS = 200_000


class WorkspaceSnapshotError(ValueError):
    """Raised when a workspace snapshot cannot be created or restored."""


@dataclass
class SnapshotFile:
    path: str
    exists: bool
    content: str = ""
    sha256: str = ""

    def to_data(self) -> dict[str, object]:
        return {
            "path": self.path,
            "exists": self.exists,
            "content": self.content,
            "sha256": self.sha256,
        }

    @classmethod
    def from_data(cls, data: object) -> "SnapshotFile | None":
        if not isinstance(data, dict):
            return None
        return cls(
            path=str(data.get("path", "")),
            exists=bool(data.get("exists", False)),
            content=str(data.get("content", "")),
            sha256=str(data.get("sha256", "")),
        )


@dataclass
class WorkspaceSnapshot:
    id: str
    project_id: str
    conversation_id: str
    message_id: str
    user_message_id: str
    created_at: str
    files: list[SnapshotFile] = field(default_factory=list)
    backup_of: str = ""

    def to_data(self) -> dict[str, object]:
        return {
            "version": 1,
            "id": self.id,
            "project_id": self.project_id,
            "conversation_id": self.conversation_id,
            "message_id": self.message_id,
            "user_message_id": self.user_message_id,
            "created_at": self.created_at,
            "backup_of": self.backup_of,
            "files": [item.to_data() for item in self.files],
        }

    @classmethod
    def from_data(cls, data: object) -> "WorkspaceSnapshot":
        if not isinstance(data, dict) or data.get("version") != 1:
            raise WorkspaceSnapshotError("Invalid workspace snapshot")
        files = [
            item
            for raw in data.get("files", [])
            if (item := SnapshotFile.from_data(raw)) is not None and item.path
        ]
        return cls(
            id=str(data.get("id", "")),
            project_id=str(data.get("project_id", "")),
            conversation_id=str(data.get("conversation_id", "")),
            message_id=str(data.get("message_id", "")),
            user_message_id=str(data.get("user_message_id", "")),
            created_at=str(data.get("created_at", "")),
            backup_of=str(data.get("backup_of", "")),
            files=files,
        )


@dataclass
class RestoreResult:
    snapshot_id: str
    backup_id: str
    restored_files: list[str]

    def to_data(self) -> dict[str, object]:
        return {
            "snapshot_id": self.snapshot_id,
            "backup_id": self.backup_id,
            "restored_files": self.restored_files,
        }


class WorkspaceSnapshotStore:
    def __init__(self, directory: Path) -> None:
        self.directory = directory
        self.directory.mkdir(parents=True, exist_ok=True)

    def capture_state(self, workspace: Workspace) -> dict[str, str]:
        state: dict[str, str] = {}
        for current_dir, dir_names, file_names in os.walk(workspace.root):
            dir_names[:] = sorted(
                name for name in dir_names if name not in PROTECTED_DIRECTORIES
            )
            current = Path(current_dir)
            for name in sorted(name for name in file_names if name not in PROTECTED_FILES):
                path = current / name
                try:
                    if path.stat().st_size > MAX_SNAPSHOT_FILE_CHARS * 4:
                        continue
                    content = path.read_text(encoding="utf-8")
                except (OSError, UnicodeDecodeError):
                    continue
                if len(content) > MAX_SNAPSHOT_FILE_CHARS:
                    continue
                state[path.relative_to(workspace.root).as_posix()] = content
        return state

    def create_for_changes(
        self,
        *,
        before: dict[str, str],
        after: dict[str, str],
        project_id: str,
        conversation_id: str,
        message_id: str,
        user_message_id: str,
        created_at: str,
    ) -> WorkspaceSnapshot | None:
        changed = sorted(path for path in set(before) | set(after) if before.get(path) != after.get(path))
        if not changed:
            return None
        snapshot = WorkspaceSnapshot(
            id=uuid4().hex,
            project_id=project_id,
            conversation_id=conversation_id,
            message_id=message_id,
            user_message_id=user_message_id,
            created_at=created_at,
            files=[_file_from_state(path, after) for path in changed],
        )
        self.save(snapshot)
        return snapshot

    def restore(self, workspace: Workspace, snapshot_id: str, *, created_at: str) -> RestoreResult:
        snapshot = self.load(snapshot_id)
        paths = [item.path for item in snapshot.files]
        backup = WorkspaceSnapshot(
            id=uuid4().hex,
            project_id=snapshot.project_id,
            conversation_id=snapshot.conversation_id,
            message_id=snapshot.message_id,
            user_message_id=snapshot.user_message_id,
            created_at=created_at,
            files=[self._current_file(workspace, path) for path in paths],
            backup_of=snapshot.id,
        )
        self.save(backup)
        restored: list[str] = []
        for item in snapshot.files:
            self._restore_file(workspace, item)
            restored.append(item.path)
        return RestoreResult(
            snapshot_id=snapshot.id,
            backup_id=backup.id,
            restored_files=restored,
        )

    def load(self, snapshot_id: str) -> WorkspaceSnapshot:
        if not snapshot_id or any(char in snapshot_id for char in "\\/."):
            raise WorkspaceSnapshotError("Invalid workspace snapshot id")
        path = self.directory / f"{snapshot_id}.json"
        if not path.exists():
            raise WorkspaceSnapshotError(f"Workspace snapshot not found: {snapshot_id}")
        try:
            return WorkspaceSnapshot.from_data(json.loads(path.read_text(encoding="utf-8")))
        except (OSError, json.JSONDecodeError) as error:
            raise WorkspaceSnapshotError(f"Cannot read workspace snapshot: {snapshot_id}") from error

    def save(self, snapshot: WorkspaceSnapshot) -> None:
        path = self.directory / f"{snapshot.id}.json"
        temporary = path.with_suffix(".tmp")
        temporary.write_text(
            json.dumps(snapshot.to_data(), ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        temporary.replace(path)

    def _current_file(self, workspace: Workspace, path: str) -> SnapshotFile:
        resolved = workspace.resolve(path)
        _reject_protected_path(workspace, resolved, path)
        if not resolved.exists():
            return SnapshotFile(path=path, exists=False)
        if not resolved.is_file():
            raise WorkspaceSnapshotError(f"Cannot snapshot non-file path: {path}")
        try:
            content = resolved.read_text(encoding="utf-8")
        except UnicodeDecodeError as error:
            raise WorkspaceSnapshotError(f"Cannot snapshot non-UTF-8 file: {path}") from error
        return SnapshotFile(
            path=path,
            exists=True,
            content=content,
            sha256=_sha256(content),
        )

    def _restore_file(self, workspace: Workspace, item: SnapshotFile) -> None:
        resolved = workspace.resolve(item.path)
        _reject_protected_path(workspace, resolved, item.path)
        if not item.exists:
            if resolved.exists():
                if not resolved.is_file():
                    raise WorkspaceSnapshotError(f"Cannot delete non-file path: {item.path}")
                resolved.unlink()
            return
        resolved.parent.mkdir(parents=True, exist_ok=True)
        temporary_path: Path | None = None
        try:
            with tempfile.NamedTemporaryFile(
                mode="w",
                encoding="utf-8",
                dir=resolved.parent,
                prefix=f".{resolved.name}.",
                suffix=".tmp",
                delete=False,
            ) as temporary:
                temporary.write(item.content)
                temporary_path = Path(temporary.name)
            os.replace(temporary_path, resolved)
        finally:
            if temporary_path is not None and temporary_path.exists():
                temporary_path.unlink()


def _file_from_state(path: str, state: dict[str, str]) -> SnapshotFile:
    if path not in state:
        return SnapshotFile(path=path, exists=False)
    content = state[path]
    return SnapshotFile(path=path, exists=True, content=content, sha256=_sha256(content))


def _reject_protected_path(workspace: Workspace, resolved_path: Path, requested_path: str) -> None:
    if workspace.is_protected(resolved_path):
        raise WorkspaceSnapshotError(f"Path is protected: {requested_path}")


def _sha256(content: str) -> str:
    return hashlib.sha256(content.encode("utf-8")).hexdigest()
