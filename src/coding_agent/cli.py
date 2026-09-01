"""Command-line interface for the coding agent."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Sequence

from coding_agent.agent import Agent, AgentError
from coding_agent.checkpoint import CheckpointStore
from coding_agent.config import ConfigError, Settings
from coding_agent.model import DeepSeekModel, ModelError
from coding_agent.project_memory import ProjectMemoryStore
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
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume the latest unfinished task for this workspace",
    )
    return parser


def _configure_stdio() -> None:
    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if not callable(reconfigure):
            continue
        try:
            reconfigure(encoding="utf-8", errors="replace")
        except (TypeError, ValueError):
            pass


def main(argv: Sequence[str] | None = None) -> int:
    _configure_stdio()
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

    def create_agent() -> Agent:
        trace = JsonlTrace.create(Path.cwd() / ".coding-agent" / "traces")
        checkpoint = CheckpointStore.for_workspace(
            Path.cwd() / ".coding-agent" / "checkpoints",
            workspace.root,
        )
        memory_store = ProjectMemoryStore(Path.cwd() / ".coding-agent" / "project-memory")
        print(f"[日志] {trace.path}")
        return Agent(
            DeepSeekModel(settings),
            workspace,
            on_event=print,
            trace=trace,
            checkpoint_store=checkpoint,
            memory_store=memory_store,
        )

    if args.resume:
        agent = create_agent()
        try:
            answer = agent.resume()
        except (AgentError, ModelError, OSError) as error:
            print(f"恢复失败: {error}", file=sys.stderr)
            return 2
        else:
            print(f"Agent > {answer}")

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
                agent = create_agent()
            try:
                answer = agent.run(task)
            except KeyboardInterrupt:
                print(
                    "\n任务已中断；可使用 --resume 从最近保存的步骤继续。",
                    file=sys.stderr,
                )
                return 130
            except (AgentError, ModelError, OSError) as error:
                print(f"任务失败: {error}", file=sys.stderr)
            else:
                print(f"Agent > {answer}")


if __name__ == "__main__":
    raise SystemExit(main())
