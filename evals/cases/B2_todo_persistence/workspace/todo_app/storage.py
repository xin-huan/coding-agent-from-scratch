"""Persistence for TODO tasks."""

from __future__ import annotations

import json
from pathlib import Path

from todo_app.core import Task


def load_tasks(path: Path) -> list[Task]:
    if not path.exists():
        return []
    return json.loads(path.read_text(encoding="utf-8"))


def save_tasks(path: Path, tasks: list[Task]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(repr(tasks) + "\n", encoding="utf-8")
