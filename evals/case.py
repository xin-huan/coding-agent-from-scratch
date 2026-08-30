"""Load fixed evaluation case definitions."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path


CASE_ORDER = {case_id: index for index, case_id in enumerate(
    ("C1", "C2", "F1", "F2", "B1", "B2", "E1", "E2", "M1", "M2", "M3")
)}


@dataclass(frozen=True)
class EvalCase:
    id: str
    category: str
    title: str
    task: str
    root: Path
    workspace: str | None
    grader: dict[str, object]
    expected_baseline: str = "passing"


def load_cases(cases_dir: Path) -> list[EvalCase]:
    cases: list[EvalCase] = []
    for manifest_path in cases_dir.glob("*/case.json"):
        data = json.loads(manifest_path.read_text(encoding="utf-8"))
        root = manifest_path.parent
        cases.append(
            EvalCase(
                id=data["id"],
                category=data["category"],
                title=data["title"],
                task=(root / data["task_file"]).read_text(encoding="utf-8"),
                root=root,
                workspace=data.get("workspace"),
                grader=data["grader"],
                expected_baseline=data.get("expected_baseline", "passing"),
            )
        )
    return sorted(cases, key=lambda case: CASE_ORDER[case.id])
