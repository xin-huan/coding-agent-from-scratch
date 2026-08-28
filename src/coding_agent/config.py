"""Application configuration."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Mapping


class ConfigError(ValueError):
    """Raised when required application configuration is invalid."""


def _read_env_file(path: Path) -> dict[str, str]:
    if not path.exists():
        return {}

    values: dict[str, str] = {}
    for line_number, raw_line in enumerate(
        path.read_text(encoding="utf-8").splitlines(), start=1
    ):
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if "=" not in line:
            raise ConfigError(f"Invalid .env entry on line {line_number}")

        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in "\"'":
            value = value[1:-1]
        values[key] = value

    return values


@dataclass(frozen=True)
class Settings:
    api_key: str = field(repr=False)
    base_url: str = "https://api.deepseek.com"
    model: str = "deepseek-v4-flash"

    @classmethod
    def load(
        cls,
        config_dir: Path,
        *,
        environ: Mapping[str, str] | None = None,
    ) -> "Settings":
        values = _read_env_file(config_dir / ".env")
        values.update(os.environ if environ is None else environ)

        api_key = values.get("DEEPSEEK_API_KEY", "").strip()
        if not api_key:
            raise ConfigError("Missing DEEPSEEK_API_KEY in the environment or .env file")

        return cls(
            api_key=api_key,
            base_url=values.get("DEEPSEEK_BASE_URL", cls.base_url).strip()
            or cls.base_url,
            model=values.get("DEEPSEEK_MODEL", cls.model).strip() or cls.model,
        )
