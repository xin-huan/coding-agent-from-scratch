"""Restricted command execution inside a project workspace."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path
from typing import Sequence

from coding_agent.workspace import Workspace


MAX_TIMEOUT_SECONDS = 60.0
MAX_OUTPUT_CHARACTERS = 20_000
PYTHON_COMMANDS = {"python", "python.exe", "python3", "python3.exe", "py", "py.exe"}
PYTEST_COMMANDS = {"pytest", "pytest.exe"}
SAFE_GIT_SUBCOMMANDS = {"status", "diff", "log", "show"}
BLOCKED_PYTHON_MODULES = {"ensurepip", "pip", "venv"}


class CommandToolError(ValueError):
    """Raised when a command violates the execution policy."""


def run_command(
    workspace: Workspace,
    argv: Sequence[str],
    *,
    cwd: str = ".",
    timeout_seconds: float = 30.0,
) -> str:
    if not argv or not all(isinstance(argument, str) for argument in argv):
        raise CommandToolError("argv must be a non-empty list of strings")
    if timeout_seconds <= 0 or timeout_seconds > MAX_TIMEOUT_SECONDS:
        raise CommandToolError(
            f"timeout_seconds must be between 0 and {MAX_TIMEOUT_SECONDS}"
        )

    working_directory = workspace.resolve(cwd)
    if workspace.is_protected(working_directory):
        raise CommandToolError(f"Working directory is protected: {cwd}")
    if not working_directory.is_dir():
        raise CommandToolError(f"Working directory does not exist: {cwd}")

    executable = Path(argv[0]).name.lower()
    arguments = list(argv[1:])
    if executable in PYTHON_COMMANDS:
        _validate_python_arguments(workspace, working_directory, arguments)
        command = [sys.executable, *arguments]
    elif executable in PYTEST_COMMANDS:
        command = [sys.executable, "-m", "pytest", *arguments]
    elif executable == "git":
        if not arguments or arguments[0] not in SAFE_GIT_SUBCOMMANDS:
            raise CommandToolError("Only read-only Git commands are allowed")
        command = ["git", *arguments]
    else:
        raise CommandToolError(f"Command is not allowed: {argv[0]}")

    try:
        completed = subprocess.run(
            command,
            cwd=working_directory,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=timeout_seconds,
            env=_safe_environment(),
            shell=False,
            check=False,
        )
    except subprocess.TimeoutExpired as error:
        raise CommandToolError(
            f"Command timed out after {timeout_seconds} seconds"
        ) from error
    except OSError as error:
        raise CommandToolError(f"Command could not start: {error}") from error

    output = [f"Exit code: {completed.returncode}"]
    if completed.stdout:
        output.extend(["STDOUT:", _truncate(completed.stdout.rstrip())])
    if completed.stderr:
        output.extend(["STDERR:", _truncate(completed.stderr.rstrip())])
    return "\n".join(output)


def _validate_python_arguments(
    workspace: Workspace,
    working_directory: Path,
    arguments: list[str],
) -> None:
    if "-c" in arguments:
        raise CommandToolError("python -c is not allowed")
    if len(arguments) >= 2 and arguments[0] == "-m":
        if arguments[1] in BLOCKED_PYTHON_MODULES:
            raise CommandToolError(f"Python module is not allowed: {arguments[1]}")
        return
    if arguments and arguments[0].endswith(".py"):
        script = workspace.resolve(working_directory / arguments[0])
        if workspace.is_protected(script):
            raise CommandToolError(f"Python script is protected: {arguments[0]}")


def _safe_environment() -> dict[str, str]:
    sensitive_names = ("KEY", "TOKEN", "SECRET", "PASSWORD", "AUTHORIZATION")
    return {
        name: value
        for name, value in os.environ.items()
        if not any(marker in name.upper() for marker in sensitive_names)
    }


def _truncate(text: str) -> str:
    if len(text) <= MAX_OUTPUT_CHARACTERS:
        return text
    return text[:MAX_OUTPUT_CHARACTERS] + "\n... output truncated ..."
