"""Tool definitions and dispatch for the coding agent."""

from __future__ import annotations

import hashlib
from typing import Mapping

from coding_agent.tools.command import run_command
from coding_agent.tools.filesystem import (
    apply_patch,
    list_files,
    read_file,
    search_text,
    write_file,
)
from coding_agent.workspace import Workspace


FILE_PATH_DESCRIPTION = (
    "File path relative to the workspace root. For example, 'README.md' or "
    "'todo_app/core.py'. Never use an absolute path or a '/workspace/...' path."
)
DIRECTORY_PATH_DESCRIPTION = (
    "Directory path relative to the workspace root. For example, '.' or "
    "'todo_app'. Never use an absolute path or a '/workspace/...' path."
)


TOOL_DEFINITIONS: list[dict[str, object]] = [
    {
        "type": "function",
        "function": {
            "name": "list_files",
            "description": "List files and directories inside the project workspace.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": DIRECTORY_PATH_DESCRIPTION,
                    }
                },
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "read_file",
            "description": (
                "Read a UTF-8 text file with line numbers. For large files, "
                "request only the relevant line range to reduce context cost."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": FILE_PATH_DESCRIPTION,
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
                },
                "required": ["path"],
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "search_text",
            "description": "Search for plain text in UTF-8 project files.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string"},
                    "path": {
                        "type": "string",
                        "description": DIRECTORY_PATH_DESCRIPTION,
                    },
                },
                "required": ["query"],
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "write_file",
            "description": "Create or fully replace a UTF-8 file in the workspace.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": FILE_PATH_DESCRIPTION,
                    },
                    "content": {"type": "string"},
                },
                "required": ["path", "content"],
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "apply_patch",
            "description": "Replace one exact block of text in an existing UTF-8 file.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": FILE_PATH_DESCRIPTION,
                    },
                    "old_text": {"type": "string"},
                    "new_text": {"type": "string"},
                },
                "required": ["path", "old_text", "new_text"],
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "run_command",
            "description": (
                "Run a command without a shell. Allowed programs are python, pytest, "
                "and read-only git status/diff/log/show. Use read_file instead of "
                "shell file viewers. Do not use cat, sed, wc, PowerShell, cmd, or "
                "python -c. Do not install dependencies."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "argv": {
                        "type": "array",
                        "items": {"type": "string"},
                    },
                    "cwd": {
                        "type": "string",
                        "description": DIRECTORY_PATH_DESCRIPTION,
                    },
                    "timeout_seconds": {
                        "type": "number",
                        "minimum": 0.1,
                        "maximum": 60,
                    },
                },
                "required": ["argv"],
                "additionalProperties": False,
            },
        },
    },
]


class ToolRegistry:
    def __init__(self, workspace: Workspace, *, cache_reads: bool = True) -> None:
        self.workspace = workspace
        self.cache_reads = cache_reads
        self._read_cache: dict[tuple[str, int | None, int | None], tuple[str, str]] = {}
        self.read_cache_hits = 0
        self.last_metadata: dict[str, object] = {}

    @property
    def definitions(self) -> list[dict[str, object]]:
        return TOOL_DEFINITIONS

    def start_task(self) -> None:
        self._read_cache.clear()
        self.read_cache_hits = 0

    def execute(self, name: str, arguments: Mapping[str, object]) -> str:
        self.last_metadata = {}
        if name == "list_files":
            path = arguments.get("path", ".")
            if not isinstance(path, str):
                raise ValueError("list_files.path must be a string")
            return list_files(self.workspace, path)

        if name == "read_file":
            path = arguments.get("path")
            start_line = arguments.get("start_line")
            end_line = arguments.get("end_line")
            if not isinstance(path, str):
                raise ValueError("read_file.path must be a string")
            if start_line is not None and not isinstance(start_line, int):
                raise ValueError("read_file.start_line must be an integer")
            if end_line is not None and not isinstance(end_line, int):
                raise ValueError("read_file.end_line must be an integer")
            cache_key = self._read_cache_key(path, start_line, end_line)
            version = self._file_version(path)
            if self.cache_reads and version:
                cached = self._cached_read(cache_key, version)
                if cached is not None:
                    self.read_cache_hits += 1
                    self.last_metadata = {
                        "path": cache_key[0],
                        "version": version,
                        "read_cache_hit": True,
                    }
                    return (
                        f"unchanged read cache hit: {cache_key[0]} "
                        f"(version {version[:12]}). Reuse the earlier content; "
                        "the full result remains in local context history."
                    )
            result = read_file(
                self.workspace,
                path,
                start_line=start_line,
                end_line=end_line,
            )
            if self.cache_reads and version:
                self._read_cache[cache_key] = (version, result)
            self.last_metadata = {
                "path": cache_key[0],
                "version": version,
                "read_cache_hit": False,
            }
            return result

        if name == "search_text":
            query = arguments.get("query")
            path = arguments.get("path", ".")
            if not isinstance(query, str) or not isinstance(path, str):
                raise ValueError("search_text query and path must be strings")
            return search_text(self.workspace, query, path)

        if name == "write_file":
            path = arguments.get("path")
            content = arguments.get("content")
            if not isinstance(path, str) or not isinstance(content, str):
                raise ValueError("write_file path and content must be strings")
            result = write_file(self.workspace, path, content)
            self._invalidate_read_cache(path)
            self.last_metadata = {
                "path": self._read_cache_key(path, None, None)[0],
                "version": self._file_version(path),
                "content": self._file_content(path),
            }
            return result

        if name == "apply_patch":
            path = arguments.get("path")
            old_text = arguments.get("old_text")
            new_text = arguments.get("new_text")
            if not isinstance(path, str):
                raise ValueError("apply_patch.path must be a string")
            if not isinstance(old_text, str) or not isinstance(new_text, str):
                raise ValueError("apply_patch arguments must be strings")
            result = apply_patch(self.workspace, path, old_text, new_text)
            self._invalidate_read_cache(path)
            self.last_metadata = {
                "path": self._read_cache_key(path, None, None)[0],
                "version": self._file_version(path),
                "content": self._file_content(path),
            }
            return result

        if name == "run_command":
            argv = arguments.get("argv")
            cwd = arguments.get("cwd", ".")
            timeout = arguments.get("timeout_seconds", 30.0)
            if not isinstance(argv, list) or not all(
                isinstance(argument, str) for argument in argv
            ):
                raise ValueError("run_command.argv must be a list of strings")
            if not isinstance(cwd, str) or not isinstance(timeout, (int, float)):
                raise ValueError("run_command cwd and timeout are invalid")
            return run_command(
                self.workspace,
                argv,
                cwd=cwd,
                timeout_seconds=float(timeout),
            )

        raise ValueError(f"Unknown tool: {name}")

    def _read_cache_key(
        self,
        path: str,
        start_line: int | None,
        end_line: int | None,
    ) -> tuple[str, int | None, int | None]:
        resolved = self.workspace.resolve(path)
        normalized = resolved.relative_to(self.workspace.root).as_posix()
        return normalized, start_line, end_line

    def _file_version(self, path: str) -> str:
        resolved = self.workspace.resolve(path)
        if self.workspace.is_protected(resolved) or not resolved.is_file():
            return ""
        return hashlib.sha256(resolved.read_bytes()).hexdigest()

    def _file_content(self, path: str) -> str:
        resolved = self.workspace.resolve(path)
        if self.workspace.is_protected(resolved) or not resolved.is_file():
            return ""
        try:
            return resolved.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            return ""

    def _cached_read(
        self,
        requested: tuple[str, int | None, int | None],
        version: str,
    ) -> tuple[str, str] | None:
        exact = self._read_cache.get(requested)
        if exact is not None and exact[0] == version:
            return exact
        return None

    def _invalidate_read_cache(self, path: str) -> None:
        normalized = self._read_cache_key(path, None, None)[0]
        self._read_cache = {
            key: value
            for key, value in self._read_cache.items()
            if key[0] != normalized
        }
