"""Local context compaction for model requests."""

from __future__ import annotations

import json
from dataclasses import dataclass


DEFAULT_MAX_CONTEXT_CHARACTERS = 28_000
DEFAULT_RECENT_TURNS = 3
MAX_RECENT_TOOL_RESULT_CHARACTERS = 6_000
MAX_SUMMARY_CHARACTERS = 6_000


@dataclass(frozen=True)
class ContextStats:
    original_characters: int
    sent_characters: int
    summarized_messages: int

    @property
    def saved_characters(self) -> int:
        return max(0, self.original_characters - self.sent_characters)


class ContextManager:
    def __init__(
        self,
        *,
        max_characters: int = DEFAULT_MAX_CONTEXT_CHARACTERS,
        recent_turns: int = DEFAULT_RECENT_TURNS,
    ) -> None:
        if max_characters < 4_000:
            raise ValueError("max_characters must be at least 4000")
        if recent_turns < 1:
            raise ValueError("recent_turns must be at least 1")
        self.max_characters = max_characters
        self.recent_turns = recent_turns
        self.last_stats = ContextStats(0, 0, 0)

    def build(
        self,
        messages: list[dict[str, object]],
        tail_messages: list[dict[str, object]] | None = None,
    ) -> list[dict[str, object]]:
        tail = tail_messages or []
        original = [*messages, *tail]
        original_size = _message_size(original)
        sanitized = [_sanitize_message(message) for message in messages]
        candidate = [*sanitized, *tail]

        if _message_size(candidate) <= self.max_characters:
            self.last_stats = ContextStats(
                original_size,
                _message_size(candidate),
                0,
            )
            return candidate

        anchor_end = 0
        for message in sanitized:
            if message.get("role") not in {"system", "user"}:
                break
            anchor_end += 1
        anchors = sanitized[:anchor_end]
        turns = _split_turns(sanitized[anchor_end:])
        old_turns = turns[:-self.recent_turns]
        recent_turns = turns[-self.recent_turns:]

        while True:
            old_messages = [message for turn in old_turns for message in turn]
            summary = _summary_message(old_messages)
            recent_messages = [message for turn in recent_turns for message in turn]
            compacted = [*anchors]
            if old_messages:
                compacted.append(summary)
            compacted.extend(recent_messages)
            compacted.extend(tail)
            if _message_size(compacted) <= self.max_characters:
                break
            if len(recent_turns) > 1:
                old_turns.append(recent_turns.pop(0))
                continue
            compacted = [
                *_limit_message_content(anchors, 2_000),
                summary,
                *_limit_message_content(recent_messages, 2_000),
                *tail,
            ]
            break

        sent_size = _message_size(compacted)
        summarized_count = sum(len(turn) for turn in old_turns)
        self.last_stats = ContextStats(
            original_size,
            sent_size,
            summarized_count,
        )
        return compacted


def _split_turns(messages: list[dict[str, object]]) -> list[list[dict[str, object]]]:
    turns: list[list[dict[str, object]]] = []
    current: list[dict[str, object]] = []
    for message in messages:
        if message.get("role") == "assistant" and current:
            turns.append(current)
            current = []
        current.append(message)
    if current:
        turns.append(current)
    return turns


def _sanitize_message(message: dict[str, object]) -> dict[str, object]:
    sanitized = dict(message)
    if message.get("role") == "tool":
        content = str(message.get("content", ""))
        sanitized["content"] = _truncate(
            content,
            MAX_RECENT_TOOL_RESULT_CHARACTERS,
            "tool output",
        )

    tool_calls = message.get("tool_calls")
    if isinstance(tool_calls, list):
        sanitized["tool_calls"] = [
            _sanitize_tool_call(call) if isinstance(call, dict) else call
            for call in tool_calls
        ]
    return sanitized


def _sanitize_tool_call(call: dict[str, object]) -> dict[str, object]:
    sanitized = dict(call)
    function = call.get("function")
    if not isinstance(function, dict):
        return sanitized
    sanitized_function = dict(function)
    arguments = function.get("arguments")
    if not isinstance(arguments, str):
        return sanitized
    try:
        parsed = json.loads(arguments)
    except json.JSONDecodeError:
        sanitized_function["arguments"] = _truncate(arguments, 1_000, "arguments")
    else:
        if isinstance(parsed, dict):
            for key in ("content", "old_text", "new_text"):
                value = parsed.get(key)
                if isinstance(value, str) and len(value) > 300:
                    parsed[key] = (
                        f"<omitted {len(value)} characters; available in workspace>"
                    )
            sanitized_function["arguments"] = json.dumps(
                parsed,
                ensure_ascii=False,
            )
    sanitized["function"] = sanitized_function
    return sanitized


def _summary_message(messages: list[dict[str, object]]) -> dict[str, object]:
    descriptions: dict[str, str] = {}
    lines = [
        "<history_summary>",
        "Older task history was compacted locally to reduce API cost.",
    ]
    for message in messages:
        role = str(message.get("role", "unknown"))
        if role == "assistant":
            content = str(message.get("content") or "").strip()
            if content:
                lines.append(f"- Assistant: {_one_line(content, 300)}")
            calls = message.get("tool_calls")
            if isinstance(calls, list):
                for call in calls:
                    if not isinstance(call, dict):
                        continue
                    call_id = str(call.get("id", "unknown"))
                    description = _tool_description(call)
                    descriptions[call_id] = description
                    lines.append(f"- Called {description}")
        elif role == "tool":
            call_id = str(message.get("tool_call_id", "unknown"))
            description = descriptions.get(call_id, f"tool {call_id}")
            content = str(message.get("content", ""))
            lines.append(f"- Result for {description}: {_result_summary(content)}")
        elif role == "system":
            lines.append(f"- Runtime note: {_one_line(str(message.get('content', '')), 300)}")
    lines.append(
        "Exact details remain in the local checkpoint and trace; re-read files when needed."
    )
    lines.append("</history_summary>")
    content = _truncate("\n".join(lines), MAX_SUMMARY_CHARACTERS, "history summary")
    return {"role": "system", "content": content}


def _tool_description(call: dict[str, object]) -> str:
    function = call.get("function")
    if not isinstance(function, dict):
        return "unknown tool"
    name = str(function.get("name", "unknown"))
    arguments = function.get("arguments")
    try:
        parsed = json.loads(arguments) if isinstance(arguments, str) else {}
    except json.JSONDecodeError:
        parsed = {}
    details: list[str] = []
    if isinstance(parsed, dict):
        for key in ("path", "query", "cwd"):
            value = parsed.get(key)
            if isinstance(value, (str, int, float, bool)):
                details.append(f"{key}={value}")
        argv = parsed.get("argv")
        if isinstance(argv, list):
            details.append("argv=" + " ".join(str(item) for item in argv[:8]))
    suffix = f" ({', '.join(details)})" if details else ""
    return f"{name}{suffix}"


def _result_summary(content: str) -> str:
    if not content:
        return "empty result"
    first_line = content.splitlines()[0]
    if content.startswith(("Exit code:", "ERROR:")):
        return _one_line(content, 800)
    if first_line.startswith(("Wrote ", "Patched ")):
        return _one_line(first_line, 500)
    return f"{_one_line(first_line, 300)}; full output omitted ({len(content)} chars)"


def _limit_message_content(
    messages: list[dict[str, object]],
    limit: int,
) -> list[dict[str, object]]:
    limited: list[dict[str, object]] = []
    for message in messages:
        copy = dict(message)
        if "content" in copy and isinstance(copy["content"], str):
            copy["content"] = _truncate(copy["content"], limit, "message")
        limited.append(copy)
    return limited


def _truncate(text: str, limit: int, label: str) -> str:
    if len(text) <= limit:
        return text
    return f"{text[:limit]}\n... {label} truncated; {len(text) - limit} chars omitted ..."


def _one_line(text: str, limit: int) -> str:
    return _truncate(" ".join(text.split()), limit, "text")


def _message_size(messages: list[dict[str, object]]) -> int:
    return len(json.dumps(messages, ensure_ascii=False, default=str))
