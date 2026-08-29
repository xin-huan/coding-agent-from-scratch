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


def _trace_counts(path: Path) -> tuple[int, int]:
    if not path.exists():
        return 0, 0
    model_calls = 0
    tool_calls = 0
    for line in path.read_text(encoding="utf-8").splitlines():
        try:
            event = json.loads(line).get("event")
        except (json.JSONDecodeError, AttributeError):
            continue
        model_calls += event == "model_request"
        tool_calls += event == "tool_start"
    return model_calls, tool_calls


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
    try:
        answer = run_agent(workspace, case.task, run_dir / "trace.jsonl")
    except Exception as error:
        answer = f"{type(error).__name__}: {error}"
        passed = False
        grader_output = "Grader was not run because the agent failed."
        failure_reason = "agent_error"
        score = 0.0
        grader_kind = case.grader.get("kind")
    else:
        grader_kind = case.grader.get("kind")
        if grader_kind == "python":
            passed, score, grader_output = _run_python_grader(case, workspace)
        elif grader_kind == "facts":
            reference = case.root / str(case.grader["reference"])
            passed, score, grader_output = grade_facts(answer, reference)
        else:
            raise ValueError(f"Unsupported grader kind: {grader_kind}")
    changed_files = _changed_files(before, _snapshot(workspace))
    changes_patch = _make_patch(before_text, _text_snapshot(workspace))
    if grader_kind == "facts" and changed_files and failure_reason != "agent_error":
        score = max(0.0, score - 10.0)
        passed = score >= 80.0
        grader_output += "Modification penalty: -10 points\n"
    if not passed and failure_reason is None:
        failure_reason = "acceptance_failed"
    model_calls, tool_calls = _trace_counts(run_dir / "trace.jsonl")

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
        model_calls=model_calls,
        tool_calls=tool_calls,
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
    lines = [
        "# Eval Report",
        "",
        f"- Cases: {len(results)}",
        f"- Passed: {passed}",
        f"- Success rate: {success_rate:.1f}%",
        f"- Average score: {average_score:.1f}",
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
            "| Case | Category | Result | Score | Seconds | Models | Tools | Failure |",
            "| --- | --- | --- | ---: | ---: | ---: | ---: | --- |",
        ]
    )
    for result in results:
        lines.append(
            f"| {result.case_id} | {result.category} | "
            f"{'PASS' if result.passed else 'FAIL'} | {result.score:.1f} | "
            f"{result.duration_seconds:.3f} | {result.model_calls} | "
            f"{result.tool_calls} | {result.failure_reason or ''} |"
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
