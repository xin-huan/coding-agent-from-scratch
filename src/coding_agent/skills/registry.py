"""Deterministic skill loading and selection."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Skill:
    name: str
    description: str
    triggers: tuple[str, ...]
    instructions: str

    def message(self) -> dict[str, object]:
        return {
            "role": "system",
            "content": (
                f"<skill name=\"{self.name}\">\n"
                f"Description: {self.description}\n\n"
                f"{self.instructions.strip()}\n"
                "</skill>"
            ),
        }


class SkillRegistry:
    def __init__(self, skills: list[Skill]) -> None:
        self.skills = skills

    @classmethod
    def load_builtin(cls) -> "SkillRegistry":
        directory = Path(__file__).with_name("builtin")
        return cls(load_skills(directory))

    def select(self, task: str, *, limit: int = 2) -> list[Skill]:
        normalized = " ".join(task.lower().split())
        scored: list[tuple[int, int, Skill]] = []
        for index, skill in enumerate(self.skills):
            score = sum(1 for trigger in skill.triggers if trigger in normalized)
            if score:
                scored.append((score, index, skill))
        scored.sort(key=lambda item: (-item[0], item[1]))
        return [skill for _score, _index, skill in scored[:limit]]


def load_skills(directory: Path) -> list[Skill]:
    if not directory.exists():
        return []
    return [
        _parse_skill_file(path)
        for path in sorted(directory.glob("*.md"))
    ]


def _parse_skill_file(path: Path) -> Skill:
    raw = path.read_text(encoding="utf-8")
    metadata: dict[str, str] = {}
    body = raw
    if raw.startswith("---\n"):
        _start, header, body = raw.split("---\n", 2)
        for line in header.splitlines():
            if ":" not in line:
                continue
            key, value = line.split(":", 1)
            metadata[key.strip()] = value.strip()
    name = metadata.get("name") or path.stem
    description = metadata.get("description", "")
    triggers = tuple(
        trigger.strip().lower()
        for trigger in metadata.get("triggers", "").split(",")
        if trigger.strip()
    )
    return Skill(
        name=name,
        description=description,
        triggers=triggers,
        instructions=body.strip(),
    )
