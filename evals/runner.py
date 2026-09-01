"""Run one evaluation case and save its observable result."""

from __future__ import annotations

import difflib
import hashlib
import json
import re
import shutil
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable

from evals.case import EvalCase
from evals.facts import grade_facts


AgentRunner = Callable[[Path, str, Path], str]


@dataclass(frozen=True)
class EvalResult:
    case_id: str
    category: str
    passed: bool
    score: float
    duration_seconds: float
    changed_files: list[str]
    failure_reason: str | None
    answer: str
    grader_output: str
    model_calls: int = 0
    tool_calls: int = 0
    agent_completed: bool = True
    acceptance_passed: bool | None = None
    prompt_tokens: int = 0
    completion_tokens: int = 0
    cache_hit_tokens: int = 0


def _snapshot(root: Path) -> dict[str, str]:
    return {
        path.relative_to(root).as_posix(): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in root.rglob("*")
        if path.is_file()
        and "__pycache__" not in path.parts
        and path.suffix not in {".pyc", ".pyo"}
    }


def _text_snapshot(root: Path) -> dict[str, str | None]:
    snapshot: dict[str, str | None] = {}
    for relative_path in _snapshot(root):
        try:
            snapshot[relative_path] = (root / relative_path).read_text(encoding="utf-8")
        except UnicodeDecodeError:
            snapshot[relative_path] = None
    return snapshot


def _make_patch(
    before: dict[str, str | None], after: dict[str, str | None]
) -> str:
    chunks: list[str] = []
    for path in sorted(before.keys() | after.keys()):
        if before.get(path) == after.get(path):
            continue
        old_text = before.get(path)
        new_text = after.get(path)
        if (old_text is None and path in before) or (
            new_text is None and path in after
        ):
            chunks.append(f"Binary file changed: {path}\n")
            continue
        chunks.extend(
            difflib.unified_diff(
                (old_text or "").splitlines(keepends=True),
                (new_text or "").splitlines(keepends=True),
                fromfile=f"a/{path}" if path in before else "/dev/null",
                tofile=f"b/{path}" if path in after else "/dev/null",
            )
        )
    return "".join(chunks)


def _changed_files(before: dict[str, str], after: dict[str, str]) -> list[str]:
    return sorted(
        path
        for path in before.keys() | after.keys()
        if before.get(path) != after.get(path)
    )


@dataclass(frozen=True)
class TraceMetrics:
    model_calls: int = 0
    tool_calls: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    cache_hit_tokens: int = 0


def _trace_metrics(path: Path) -> TraceMetrics:
    if not path.exists():
        return TraceMetrics()
    model_calls = 0
    tool_calls = 0
    prompt_tokens = 0
    completion_tokens = 0
    cache_hit_tokens = 0
    for line in path.read_text(encoding="utf-8").splitlines():
        try:
            record = json.loads(line)
            event = record.get("event")
            data = record.get("data", {})
        except (json.JSONDecodeError, AttributeError):
            continue
        model_calls += event == "model_request"
        tool_calls += event == "tool_start"
        if not isinstance(data, dict):
            continue
        if event == "token_usage":
            prompt_tokens += _integer_metric(data.get("prompt_tokens"))
            completion_tokens += _integer_metric(data.get("completion_tokens"))
            cache_hit_tokens += _integer_metric(data.get("cache_hit_tokens"))
    return TraceMetrics(
        model_calls=model_calls,
        tool_calls=tool_calls,
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        cache_hit_tokens=cache_hit_tokens,
    )


def _integer_metric(value: object) -> int:
    return value if isinstance(value, int) and value >= 0 else 0


def _prepare_workspace(case: EvalCase, workspace: Path) -> None:
    if case.workspace is None:
        workspace.mkdir()
        return
    shutil.copytree(case.root / case.workspace, workspace)


def _unittest_score(output: str, passed: bool) -> float:
    if passed:
        return 100.0
    match = re.search(r"Ran\s+(\d+)\s+tests?", output)
    if not match:
        return 0.0
    total = int(match.group(1))
    failed = sum(
        int(count)
        for _kind, count in re.findall(r"(failures|errors)=(\d+)", output)
    )
    return round(100 * max(total - failed, 0) / total, 1) if total else 0.0


def _run_python_grader(case: EvalCase, workspace: Path) -> tuple[bool, float, str]:
    grader = case.root / str(case.grader["path"])
    timeout = float(case.grader.get("timeout", 20))
    try:
        completed = subprocess.run(
            [sys.executable, str(grader), str(workspace)],
            cwd=case.root,
            capture_output=True,
            text=True,
            timeout=timeout,
            check=False,
        )
    except subprocess.TimeoutExpired as error:
        output = (error.stdout or "") + (error.stderr or "")
        return False, 0.0, f"Grader timed out after {timeout:g}s\n{output}"
    output = completed.stdout + completed.stderr
    passed = completed.returncode == 0
    return passed, _unittest_score(output, passed), output


def evaluate_case(
    case: EvalCase,
    run_agent: AgentRunner,
    run_dir: Path,
) -> EvalResult:
    run_dir.mkdir(parents=True, exist_ok=False)
    workspace = run_dir / "workspace"
    _prepare_workspace(case, workspace)
    before = _snapshot(workspace)
    before_text = _text_snapshot(workspace)
    (run_dir / "task.md").write_text(case.task, encoding="utf-8")

    started = time.monotonic()
    failure_reason: str | None = None
    agent_completed = True
    try:
        answer = run_agent(workspace, case.task, run_dir / "trace.jsonl")
    except Exception as error:
        agent_completed = False
        answer = f"{type(error).__name__}: {error}"
        failure_reason = "agent_error"
    grader_kind = case.grader.get("kind")
    if grader_kind == "python":
        acceptance_passed, score, grader_output = _run_python_grader(case, workspace)
    elif grader_kind == "facts":
        reference = case.root / str(case.grader["reference"])
        acceptance_passed, score, grader_output = grade_facts(answer, reference)
    else:
        raise ValueError(f"Unsupported grader kind: {grader_kind}")
    changed_files = _changed_files(before, _snapshot(workspace))
    changes_patch = _make_patch(before_text, _text_snapshot(workspace))
    if grader_kind == "facts" and changed_files and agent_completed:
        score = max(0.0, score - 10.0)
        acceptance_passed = score >= 80.0
        grader_output += "Modification penalty: -10 points\n"
    passed = agent_completed and acceptance_passed
    if not acceptance_passed and failure_reason is None:
        failure_reason = "acceptance_failed"
    metrics = _trace_metrics(run_dir / "trace.jsonl")

    result = EvalResult(
        case_id=case.id,
        category=case.category,
        passed=passed,
        score=score,
        duration_seconds=round(time.monotonic() - started, 3),
        changed_files=changed_files,
        failure_reason=failure_reason,
        answer=answer,
        grader_output=grader_output,
        model_calls=metrics.model_calls,
        tool_calls=metrics.tool_calls,
        agent_completed=agent_completed,
        acceptance_passed=acceptance_passed,
        prompt_tokens=metrics.prompt_tokens,
        completion_tokens=metrics.completion_tokens,
        cache_hit_tokens=metrics.cache_hit_tokens,
    )
    (run_dir / "answer.txt").write_text(answer, encoding="utf-8")
    (run_dir / "grader.txt").write_text(grader_output, encoding="utf-8")
    (run_dir / "changes.patch").write_text(changes_patch, encoding="utf-8")
    (run_dir / "result.json").write_text(
        json.dumps(asdict(result), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return result


def write_summary(results: list[EvalResult], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    payload = [asdict(result) for result in results]
    (output_dir / "results.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    passed = sum(result.passed for result in results)
    success_rate = 100 * passed / len(results) if results else 0.0
    average_score = (
        sum(result.score for result in results) / len(results) if results else 0.0
    )
    prompt_tokens = sum(result.prompt_tokens for result in results)
    completion_tokens = sum(result.completion_tokens for result in results)
    cache_hit_tokens = sum(result.cache_hit_tokens for result in results)
    lines = [
        "# Eval Report",
        "",
        f"- Cases: {len(results)}",
        f"- Passed: {passed}",
        f"- Success rate: {success_rate:.1f}%",
        f"- Average score: {average_score:.1f}",
        f"- Prompt tokens: {prompt_tokens}",
        f"- Completion tokens: {completion_tokens}",
        f"- Cache-hit tokens: {cache_hit_tokens}",
        "",
        "## Categories",
        "",
        "| Category | Cases | Passed | Success rate |",
        "| --- | ---: | ---: | ---: |",
    ]
    categories = dict.fromkeys(result.category for result in results)
    for category in categories:
        category_results = [result for result in results if result.category == category]
        category_passed = sum(result.passed for result in category_results)
        category_rate = 100 * category_passed / len(category_results)
        lines.append(
            f"| {category} | {len(category_results)} | {category_passed} | "
            f"{category_rate:.1f}% |"
        )
    lines.extend(
        [
            "",
            "## Cases",
            "",
            "| Case | Category | Result | Agent | Acceptance | Score | Seconds | Models | Tools | Input tok | Output tok | Failure |",
            "| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |",
        ]
    )
    for result in results:
        lines.append(
            f"| {result.case_id} | {result.category} | "
            f"{'PASS' if result.passed else 'FAIL'} | "
            f"{'yes' if result.agent_completed else 'no'} | "
            f"{'yes' if result.acceptance_passed else 'no'} | {result.score:.1f} | "
            f"{result.duration_seconds:.3f} | {result.model_calls} | "
            f"{result.tool_calls} | {result.prompt_tokens} | "
            f"{result.completion_tokens} | "
            f"{result.failure_reason or ''} |"
        )
    (output_dir / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    failure_lines = []
    for result in results:
        if result.passed:
            continue
        failure_lines.append(
            json.dumps(
                {
                    "case_id": result.case_id,
                    "category": result.category,
                    "failure_type": result.failure_reason,
                    "score": result.score,
                    "changed_files": result.changed_files,
                    "model_calls": result.model_calls,
                    "tool_calls": result.tool_calls,
                    "prompt_tokens": result.prompt_tokens,
                    "completion_tokens": result.completion_tokens,
                    "agent_completed": result.agent_completed,
                    "acceptance_passed": result.acceptance_passed,
                    "diagnosis": "",
                    "follow_up": "",
                },
                ensure_ascii=False,
            )
        )
    content = "\n".join(failure_lines)
    (output_dir / "failure_cases.jsonl").write_text(
        content + ("\n" if content else ""),
        encoding="utf-8",
    )
