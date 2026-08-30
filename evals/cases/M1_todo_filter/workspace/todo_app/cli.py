"""Command parsing and terminal output."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Sequence

from todo_app.repository import TaskRepository
from todo_app.service import TodoService


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Layered TODO")
    parser.add_argument("--data", type=Path, default=Path("tasks.json"))
    commands = parser.add_subparsers(dest="command", required=True)
    add = commands.add_parser("add")
    add.add_argument("title")
    commands.add_parser("list")
    args = parser.parse_args(argv)

    service = TodoService(TaskRepository(args.data))
    if args.command == "add":
        service.add(args.title)
    else:
        for task in service.list_tasks():
            print(f"{task['id']}: {task['title']}")
    return 0
