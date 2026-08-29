"""Command-line interface for the coding agent."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Sequence

from coding_agent.agent import Agent, AgentError
from coding_agent.config import ConfigError, Settings
from coding_agent.model import DeepSeekModel, ModelError
from coding_agent.trace import JsonlTrace
from coding_agent.workspace import Workspace, WorkspaceError


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
    try:
        workspace = Workspace(args.workspace)
    except WorkspaceError as error:
        print(f"工作目录错误: {error}", file=sys.stderr)
        return 2

    try:
        settings = Settings.load(Path.cwd())
    except ConfigError as error:
        print(f"配置错误: {error}", file=sys.stderr)
        return 2

    print("Coding Agent")
    print(f"工作目录: {workspace.root}")
    print(f"模型: {settings.model}")
    agent: Agent | None = None

    while True:
        try:
            task = input("你 > ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            return 0

        if task == "/exit":
            return 0
        if task:
            if agent is None:
                trace = JsonlTrace.create(Path.cwd() / ".coding-agent" / "traces")
                print(f"[日志] {trace.path}")
                agent = Agent(
                    DeepSeekModel(settings),
                    workspace,
                    on_event=print,
                    trace=trace,
                )
            try:
                answer = agent.run(task)
            except (AgentError, ModelError, OSError) as error:
                print(f"任务失败: {error}", file=sys.stderr)
            else:
                print(f"Agent > {answer}")


if __name__ == "__main__":
    raise SystemExit(main())
