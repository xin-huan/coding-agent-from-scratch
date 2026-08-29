"""Safe access to a single project workspace."""

from __future__ import annotations

from pathlib import Path


class WorkspaceError(ValueError):
    """Raised when a path cannot be used inside the workspace."""


class Workspace:
    def __init__(self, root: Path) -> None:
        self.root = root.resolve()
        if not self.root.is_dir():
            raise WorkspaceError(f"Workspace does not exist: {self.root}")

    def resolve(self, relative_path: str | Path) -> Path:
        resolved = (self.root / relative_path).resolve()
        try:
            resolved.relative_to(self.root)
        except ValueError as error:
            raise WorkspaceError(f"Path is outside workspace: {relative_path}") from error
        return resolved
