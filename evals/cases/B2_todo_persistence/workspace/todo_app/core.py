"""TODO business rules."""

from __future__ import annotations


Task = dict[str, object]


def add_task(tasks: list[Task], title: str) -> list[Task]:
    title = title.strip()
    if not title:
        raise ValueError("Task title cannot be empty")
    next_id = max((int(task["id"]) for task in tasks), default=0) + 1
    return [*tasks, {"id": next_id, "title": title, "done": False}]
