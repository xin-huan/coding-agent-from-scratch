"""JSON task storage."""

from __future__ import annotations

import json
from pathlib import Path

from todo_app.models import Task


class TaskRepository:
    def __init__(self, path: Path) -> None:
        self.path = path

    def load(self) -> list[Task]:
        if not self.path.exists():
            return []
        return json.loads(self.path.read_text(encoding="utf-8"))

    def save(self, tasks: list[Task]) -> None:
        self.path.write_text(json.dumps(tasks, indent=2) + "\n", encoding="utf-8")
