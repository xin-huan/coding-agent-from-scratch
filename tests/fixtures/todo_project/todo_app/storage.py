"""JSON persistence for TODO tasks."""

from __future__ import annotations

import json
from pathlib import Path

from todo_app.core import Task


def load_tasks(path: Path) -> list[Task]:
    if not path.exists():
        return []
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, list):
        raise ValueError("Task file must contain a JSON list")
    return data


def save_tasks(path: Path, tasks: list[Task]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(tasks, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

