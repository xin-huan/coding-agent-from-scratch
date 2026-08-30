"""Local filesystem tools."""

from __future__ import annotations

import os
import tempfile
from pathlib import Path

from coding_agent.workspace import PROTECTED_DIRECTORIES, PROTECTED_FILES, Workspace


IGNORED_DIRECTORIES = PROTECTED_DIRECTORIES
MAX_WRITE_CHARACTERS = 100_000
MAX_READ_CHARACTERS = 20_000


class FileToolError(ValueError):
    """Raised when a filesystem tool cannot complete its request."""


def list_files(workspace: Workspace, path: str = ".") -> str:
    directory = workspace.resolve(path)
    _reject_protected_path(workspace, directory, path)
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
        for name in sorted(name for name in file_names if name not in PROTECTED_FILES):
            relative = (current / name).relative_to(workspace.root).as_posix()
            entries.append(relative)

    return "\n".join(sorted(entries)) or "(empty directory)"


def read_file(
    workspace: Workspace,
    path: str,
    start_line: int | None = None,
    end_line: int | None = None,
) -> str:
    file_path = workspace.resolve(path)
    _reject_protected_path(workspace, file_path, path)
    if not file_path.is_file():
        raise FileToolError(f"File does not exist: {path}")

    try:
        lines = file_path.read_text(encoding="utf-8").splitlines()
    except UnicodeDecodeError as error:
        raise FileToolError(f"File is not valid UTF-8 text: {path}") from error

    if not lines:
        return "(empty file)"
    start = 1 if start_line is None else start_line
    end = len(lines) if end_line is None else min(end_line, len(lines))
    if start < 1 or end < start:
        raise FileToolError("read_file line range is invalid")
    if start > len(lines):
        raise FileToolError(f"start_line exceeds file length: {len(lines)}")

    rendered = [
        f"{number} | {lines[number - 1]}"
        for number in range(start, end + 1)
    ]
    result = "\n".join(rendered)
    if len(result) <= MAX_READ_CHARACTERS:
        return result

    truncated = result[:MAX_READ_CHARACTERS]
    if "\n" not in truncated:
        return (
            f"{truncated}\n... line {start} output truncated; "
            "use search_text to locate a smaller relevant region ..."
        )
    last_complete_line = truncated.count("\n") + start
    return (
        f"{truncated}\n... file output truncated; "
        f"continue with start_line={last_complete_line} ..."
    )


def search_text(
    workspace: Workspace,
    query: str,
    path: str = ".",
    *,
    max_results: int = 100,
) -> str:
    if not query:
        raise FileToolError("Search query cannot be empty")

    directory = workspace.resolve(path)
    _reject_protected_path(workspace, directory, path)
    if not directory.is_dir():
        raise FileToolError(f"Directory does not exist: {path}")

    matches: list[str] = []
    for current_dir, dir_names, file_names in os.walk(directory):
        dir_names[:] = [
            name for name in dir_names if name not in IGNORED_DIRECTORIES
        ]
        current = Path(current_dir)
        for name in file_names:
            if name in PROTECTED_FILES:
                continue
            file_path = current / name
            try:
                lines = file_path.read_text(encoding="utf-8").splitlines()
            except (OSError, UnicodeDecodeError):
                continue
            relative = file_path.relative_to(workspace.root).as_posix()
            for line_number, line in enumerate(lines, start=1):
                if query in line:
                    matches.append(f"{relative}:{line_number}: {line.strip()}")
                    if len(matches) >= max_results:
                        return "\n".join(matches)

    return "\n".join(matches) or "(no matches)"


def write_file(workspace: Workspace, path: str, content: str) -> str:
    if len(content) > MAX_WRITE_CHARACTERS:
        raise FileToolError(
            f"Content exceeds the {MAX_WRITE_CHARACTERS} character limit"
        )

    file_path = workspace.resolve(path)
    _reject_protected_path(workspace, file_path, path)
    file_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=file_path.parent,
            prefix=f".{file_path.name}.",
            suffix=".tmp",
            delete=False,
        ) as temporary:
            temporary.write(content)
            temporary_path = Path(temporary.name)
        os.replace(temporary_path, file_path)
    finally:
        if temporary_path is not None and temporary_path.exists():
            temporary_path.unlink()

    return f"Wrote {path} ({len(content)} characters)"


def apply_patch(
    workspace: Workspace,
    path: str,
    old_text: str,
    new_text: str,
) -> str:
    if not old_text:
        raise FileToolError("old_text cannot be empty")

    file_path = workspace.resolve(path)
    _reject_protected_path(workspace, file_path, path)
    if not file_path.is_file():
        raise FileToolError(f"File does not exist: {path}")

    try:
        content = file_path.read_text(encoding="utf-8")
    except UnicodeDecodeError as error:
        raise FileToolError(f"File is not valid UTF-8 text: {path}") from error

    occurrences = content.count(old_text)
    if occurrences != 1:
        raise FileToolError(
            f"old_text must appear exactly once in {path}; found {occurrences}"
        )

    updated = content.replace(old_text, new_text, 1)
    write_file(workspace, path, updated)
    return f"Patched {path}"


def _reject_protected_path(
    workspace: Workspace,
    resolved_path: Path,
    requested_path: str,
) -> None:
    if workspace.is_protected(resolved_path):
        raise FileToolError(f"Path is protected: {requested_path}")
