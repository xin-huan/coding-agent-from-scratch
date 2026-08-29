"""Tool definitions and dispatch for the coding agent."""

from __future__ import annotations

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
                        "description": "Workspace-relative directory path.",
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
            "description": "Read a UTF-8 text file with line numbers.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "Workspace-relative file path.",
                    }
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
                        "description": "Workspace-relative directory path.",
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
                    "path": {"type": "string"},
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
                    "path": {"type": "string"},
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
            "description": "Run an approved command in the workspace without a shell.",
            "parameters": {
                "type": "object",
                "properties": {
                    "argv": {
                        "type": "array",
                        "items": {"type": "string"},
                    },
                    "cwd": {
                        "type": "string",
                        "description": "Workspace-relative working directory.",
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
    def __init__(self, workspace: Workspace) -> None:
        self.workspace = workspace

    @property
    def definitions(self) -> list[dict[str, object]]:
        return TOOL_DEFINITIONS

    def execute(self, name: str, arguments: Mapping[str, object]) -> str:
        if name == "list_files":
            path = arguments.get("path", ".")
            if not isinstance(path, str):
                raise ValueError("list_files.path must be a string")
            return list_files(self.workspace, path)

        if name == "read_file":
            path = arguments.get("path")
            if not isinstance(path, str):
                raise ValueError("read_file.path must be a string")
            return read_file(self.workspace, path)

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
            return write_file(self.workspace, path, content)

        if name == "apply_patch":
            path = arguments.get("path")
            old_text = arguments.get("old_text")
            new_text = arguments.get("new_text")
            if not isinstance(path, str):
                raise ValueError("apply_patch.path must be a string")
            if not isinstance(old_text, str) or not isinstance(new_text, str):
                raise ValueError("apply_patch arguments must be strings")
            return apply_patch(self.workspace, path, old_text, new_text)

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
