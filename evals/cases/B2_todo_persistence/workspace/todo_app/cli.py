"""TODO command-line interface."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Sequence

from todo_app.core import add_task
from todo_app.storage import load_tasks, save_tasks


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Persistent TODO")
    parser.add_argument("--data", type=Path, default=Path("tasks.json"))
    commands = parser.add_subparsers(dest="command", required=True)
    add = commands.add_parser("add")
    add.add_argument("title")
    commands.add_parser("list")
    args = parser.parse_args(argv)

    tasks = load_tasks(args.data)
    if args.command == "add":
        tasks = add_task(tasks, args.title)
        save_tasks(args.data, tasks)
    else:
        for task in tasks:
            print(f"{task['id']}: {task['title']}")
    return 0
