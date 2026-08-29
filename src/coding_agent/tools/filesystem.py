"""Read-only filesystem tools."""

from __future__ import annotations

import os
from pathlib import Path

from coding_agent.workspace import Workspace


IGNORED_DIRECTORIES = {".git", ".venv", "__pycache__"}


class FileToolError(ValueError):
    """Raised when a filesystem tool cannot complete its request."""


def list_files(workspace: Workspace, path: str = ".") -> str:
    directory = workspace.resolve(path)
    if not directory.is_dir():
        raise FileToolError(f"Directory does not exist: {path}")

    entries: list[str] = []
    for current_dir, dir_names, file_names in os.walk(directory):
        dir_names[:] = sorted(
            name for name in dir_names if name not in IGNORED_DIRECTORIES
        )
        current = Path(current_dir)

        for name in dir_names:
            relative = (current / name).relative_to(workspace.root).as_posix()
            entries.append(f"{relative}/")
        for name in sorted(file_names):
            relative = (current / name).relative_to(workspace.root).as_posix()
            entries.append(relative)

    return "\n".join(sorted(entries)) or "(empty directory)"


def read_file(workspace: Workspace, path: str) -> str:
    file_path = workspace.resolve(path)
    if not file_path.is_file():
        raise FileToolError(f"File does not exist: {path}")

    try:
        lines = file_path.read_text(encoding="utf-8").splitlines()
    except UnicodeDecodeError as error:
        raise FileToolError(f"File is not valid UTF-8 text: {path}") from error

    if not lines:
        return "(empty file)"
    return "\n".join(f"{number} | {line}" for number, line in enumerate(lines, start=1))
