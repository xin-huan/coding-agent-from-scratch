"""JSON note storage."""

from __future__ import annotations

import json
from pathlib import Path


class NoteRepository:
    def __init__(self, path: Path) -> None:
        self.path = path

    def load(self) -> list[dict[str, object]]:
        if not self.path.exists():
            return []
        return json.loads(self.path.read_text(encoding="utf-8"))

    def save(self, notes: list[dict[str, object]]) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps(notes, indent=2) + "\n", encoding="utf-8")
