"""Deterministic scoring for project-explanation answers."""

from __future__ import annotations

import json
import re
from pathlib import Path


SECTION = re.compile(r"^\s*(\d+)\s*[.、)]\s*(.*)$")


def _normalise(text: str) -> str:
    return text.casefold().replace("\\", "/")


def _split_sections(answer: str) -> dict[int, str]:
    sections: dict[int, list[str]] = {}
    current: int | None = None
    for line in answer.splitlines():
        match = SECTION.match(line)
        if match:
            current = int(match.group(1))
            sections[current] = [match.group(2)]
        elif current is not None:
            sections[current].append(line)
    return {number: "\n".join(lines) for number, lines in sections.items()}


def grade_facts(answer: str, reference_path: Path) -> tuple[bool, float, str]:
    reference = json.loads(reference_path.read_text(encoding="utf-8"))
    questions = reference["questions"]
    sections = _split_sections(answer)
    details: list[dict[str, object]] = []
    earned = 0.0

    for question in questions:
        question_id = int(question["id"])
        expected = [str(term) for term in question["all_of"]]
        section = _normalise(sections.get(question_id, ""))
        matched = [term for term in expected if _normalise(term) in section]
        fraction = len(matched) / len(expected) if expected else 1.0
        earned += fraction
        details.append(
            {
                "question": question_id,
                "matched": matched,
                "expected_count": len(expected),
                "score": round(fraction, 3),
            }
        )

    score = round(100 * earned / len(questions), 1) if questions else 0.0
    output = json.dumps({"questions": details, "score": score}, ensure_ascii=False, indent=2)
    return score >= 80.0, score, output + "\n"
