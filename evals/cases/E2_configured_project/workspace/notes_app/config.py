"""Configuration loading with explicit precedence."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Settings:
    data_path: Path

    @classmethod
    def load(cls, config_path: Path = Path("config.json")) -> "Settings":
        data_path = "notes.json"
        if config_path.exists():
            values = json.loads(config_path.read_text(encoding="utf-8"))
            data_path = values.get("data_path", data_path)
        data_path = os.environ.get("NOTES_DATA_PATH", data_path)
        return cls(data_path=Path(data_path))
