"""Command-line entry point for fixed business evaluations."""

from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path
from typing import Sequence

from coding_agent.agent import DEFAULT_MAX_STEPS, Agent
from coding_agent.config import ConfigError, Settings
from coding_agent.model import DeepSeekModel
from coding_agent.trace import JsonlTrace
from coding_agent.workspace import Workspace
from evals.case import EvalCase, load_cases
from evals.runner import AgentRunner, EvalResult, evaluate_case, write_summary


PROJECT_ROOT = Path(__file__).parent.parent
CASES_DIR = Path(__file__).parent / "cases"
DEFAULT_RESULTS_DIR = Path(__file__).parent / "results"


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run fixed Coding Agent evaluations")
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument("--list", action="store_true", help="List cases without calling a model")
    action.add_argument(
        "--run",
        nargs="+",
        metavar="CASE",
        help="Run case IDs, or 'all', using the configured model",
    )
    parser.add_argument("--repeat", type=int, default=1, help="Runs per selected case")
    parser.add_argument("--max-steps", type=int, default=DEFAULT_MAX_STEPS)
    parser.add_argument("--output", type=Path, default=DEFAULT_RESULTS_DIR)
    return parser


def _select_cases(parser: argparse.ArgumentParser, requested: list[str]) -> list[EvalCase]:
    cases = load_cases(CASES_DIR)
    if requested == ["all"]:
        return cases
    by_id = {case.id: case for case in cases}
    unknown = [case_id for case_id in requested if case_id not in by_id]
    if unknown:
        parser.error(f"Unknown case: {', '.join(unknown)}")
    return [by_id[case_id] for case_id in requested]


def _real_agent_runner(
    settings: Settings,
    max_steps: int,
) -> AgentRunner:
    model = DeepSeekModel(settings)

    def run(workspace: Path, task: str, trace_path: Path) -> str:
        agent = Agent(
            model,
            Workspace(workspace),
            max_steps=max_steps,
            on_event=print,
            trace=JsonlTrace(trace_path),
        )
        return agent.run(task)

    return run


def main(argv: Sequence[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    cases = load_cases(CASES_DIR)
    if args.list:
        for case in cases:
            print(f"{case.id}  {case.category:<7}  {case.title}")
        return 0

    if args.repeat < 1 or args.max_steps < 1:
        parser.error("--repeat and --max-steps must be at least 1")
    selected = _select_cases(parser, args.run)
    try:
        settings = Settings.load(PROJECT_ROOT)
    except ConfigError as error:
        parser.error(str(error))

    session_name = datetime.now().strftime("%Y%m%d-%H%M%S-%f")
    session_dir = args.output.resolve() / session_name
    session_dir.mkdir(parents=True)
    run_agent = _real_agent_runner(settings, args.max_steps)
    results: list[EvalResult] = []

    for case in selected:
        for run_number in range(1, args.repeat + 1):
            print(f"[{case.id}] run {run_number}/{args.repeat}: {case.title}")
            run_dir = session_dir / f"{case.id}-run-{run_number}"
            result = evaluate_case(case, run_agent, run_dir)
            results.append(result)
            print(f"[{case.id}] {'PASS' if result.passed else 'FAIL'} ({result.score:.1f})")

    write_summary(results, session_dir)
    print(f"Report: {session_dir / 'report.md'}")
    return 0 if all(result.passed for result in results) else 1
