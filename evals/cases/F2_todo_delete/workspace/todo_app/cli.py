"""TODO command-line interface."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Sequence

from todo_app.core import add_task, complete_task
from todo_app.storage import load_tasks, save_tasks


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Mini TODO")
    parser.add_argument("--data", type=Path, default=Path("tasks.json"))
    commands = parser.add_subparsers(dest="command", required=True)
    add = commands.add_parser("add")
    add.add_argument("title")
    commands.add_parser("list")
    done = commands.add_parser("done")
    done.add_argument("task_id", type=int)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    tasks = load_tasks(args.data)
    try:
        if args.command == "add":
            tasks = add_task(tasks, args.title)
            save_tasks(args.data, tasks)
        elif args.command == "done":
            tasks = complete_task(tasks, args.task_id)
            save_tasks(args.data, tasks)
        else:
            for task in tasks:
                marker = "x" if task["done"] else " "
                print(f"[{marker}] {task['id']}: {task['title']}")
    except ValueError as error:
        parser.error(str(error))
    return 0
