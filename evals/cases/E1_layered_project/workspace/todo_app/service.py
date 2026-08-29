"""TODO use cases."""

from __future__ import annotations

from todo_app.models import Task
from todo_app.repository import TaskRepository


class TodoService:
    def __init__(self, repository: TaskRepository) -> None:
        self.repository = repository

    def add(self, title: str) -> Task:
        tasks = self.repository.load()
        task: Task = {"id": len(tasks) + 1, "title": title.strip(), "done": False}
        tasks.append(task)
        self.repository.save(tasks)
        return task

    def list_tasks(self) -> list[Task]:
        return self.repository.load()
