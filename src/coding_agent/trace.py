"""Structured local execution traces."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Protocol
from uuid import uuid4


SENSITIVE_MARKERS = ("key", "token", "secret", "password", "authorization")
SAFE_METRIC_FIELDS = {
    "prompt_tokens",
    "completion_tokens",
    "total_tokens",
    "cache_hit_tokens",
    "cumulative_prompt_tokens",
    "cumulative_completion_tokens",
    "cumulative_cache_hit_tokens",
    "original_tokens",
    "sent_tokens",
    "saved_tokens",
    "tool_definition_tokens",
}


class Trace(Protocol):
    def record(self, event: str, **data: object) -> None: ...


class NullTrace:
    def record(self, event: str, **data: object) -> None:
        return None


class JsonlTrace:
    def __init__(self, path: Path) -> None:
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)

    @classmethod
    def create(cls, directory: Path) -> "JsonlTrace":
        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        return cls(directory / f"{timestamp}-{uuid4().hex[:8]}.jsonl")

    def record(self, event: str, **data: object) -> None:
        record = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "event": event,
            "data": _sanitize(data),
        }
        with self.path.open("a", encoding="utf-8") as output:
            output.write(json.dumps(record, ensure_ascii=False) + "\n")


def _sanitize(value: Any, field_name: str = "") -> Any:
    normalized_field = field_name.lower()
    if normalized_field not in SAFE_METRIC_FIELDS and any(
        marker in normalized_field for marker in SENSITIVE_MARKERS
    ):
        return "[REDACTED]"
    if isinstance(value, dict):
        return {key: _sanitize(item, str(key)) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_sanitize(item) for item in value]
    return value
