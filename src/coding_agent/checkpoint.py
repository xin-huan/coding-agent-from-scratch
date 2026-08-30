"""Atomic local checkpoints for resumable agent runs."""

from __future__ import annotations

import json
import os
from hashlib import sha256
from pathlib import Path
from uuid import uuid4


class CheckpointError(ValueError):
    """Raised when a checkpoint is missing or invalid."""


class CheckpointStore:
    def __init__(self, path: Path) -> None:
        self.path = path

    @classmethod
    def for_workspace(cls, directory: Path, workspace: Path) -> "CheckpointStore":
        identity = str(workspace.resolve()).casefold().encode("utf-8")
        name = f"{sha256(identity).hexdigest()[:16]}.json"
        return cls(directory / name)

    def save(self, data: dict[str, object]) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        temporary = self.path.with_name(f"{self.path.name}.{uuid4().hex}.tmp")
        try:
            with temporary.open("w", encoding="utf-8") as output:
                json.dump(data, output, ensure_ascii=False, indent=2)
                output.write("\n")
                output.flush()
                os.fsync(output.fileno())
            os.replace(temporary, self.path)
        finally:
            temporary.unlink(missing_ok=True)

    def load(self) -> dict[str, object]:
        if not self.path.exists():
            raise CheckpointError("No unfinished task checkpoint was found")
        try:
            data = json.loads(self.path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as error:
            raise CheckpointError(f"Invalid checkpoint: {error}") from error
        if not isinstance(data, dict):
            raise CheckpointError("Invalid checkpoint: root must be an object")
        return data

    def clear(self) -> None:
        self.path.unlink(missing_ok=True)
