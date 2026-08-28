"""Command-line interface for the coding agent."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Sequence

from coding_agent.config import ConfigError, Settings


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="A lightweight local coding agent")
    parser.add_argument(
        "--workspace",
        type=Path,
        default=Path.cwd(),
        help="Project directory the agent may access",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    workspace = args.workspace.resolve()
    if not workspace.is_dir():
        print(f"工作目录不存在: {workspace}", file=sys.stderr)
        return 2

    try:
        settings = Settings.load(Path.cwd())
    except ConfigError as error:
        print(f"配置错误: {error}", file=sys.stderr)
        return 2

    print("Coding Agent")
    print(f"工作目录: {workspace}")
    print(f"模型: {settings.model}")

    while True:
        try:
            task = input("你 > ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            return 0

        if task == "/exit":
            return 0
        if task:
            print("Agent Runtime 将在下一阶段接入。")


if __name__ == "__main__":
    raise SystemExit(main())

