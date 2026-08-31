"""Token-budgeted context assembly for model requests."""

from __future__ import annotations

import ast
import json
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping


DEFAULT_MAX_PROMPT_TOKENS = 6_000
DEFAULT_RECENT_TURNS = 2
MAX_WORKING_RESULT_TOKENS = 800


@dataclass(frozen=True)
class ContextStats:
    original_characters: int = 0
    sent_characters: int = 0
    original_tokens: int = 0
    sent_tokens: int = 0
    tool_definition_tokens: int = 0
    summarized_messages: int = 0
    retrieved_entries: int = 0

    @property
    def saved_characters(self) -> int:
        return max(0, self.original_characters - self.sent_characters)

    @property
    def saved_tokens(self) -> int:
        return max(0, self.original_tokens - self.sent_tokens)


@dataclass(frozen=True)
class HistoryEntry:
    step: int
    kind: str
    name: str
    path: str
    version: str
    summary: str
    content: str

    def to_data(self) -> dict[str, object]:
        return {
            "step": self.step,
            "kind": self.kind,
            "name": self.name,
            "path": self.path,
            "version": self.version,
            "summary": self.summary,
            "content": self.content,
        }

    @classmethod
    def from_data(cls, value: object) -> "HistoryEntry | None":
        if not isinstance(value, dict):
            return None
        step = value.get("step")
        if not isinstance(step, int):
            return None
        return cls(
            step=step,
            kind=str(value.get("kind", "unknown")),
            name=str(value.get("name", "unknown")),
            path=str(value.get("path", "")),
            version=str(value.get("version", "")),
            summary=str(value.get("summary", "")),
            content=str(value.get("content", "")),
        )


class ContextArchive:
    """Append-only task history with lightweight lexical retrieval."""

    def __init__(self, path: Path | None = None) -> None:
        self.path = path
        self.entries: list[HistoryEntry] = []
        self._load()

    def reset(self) -> None:
        self.entries.clear()
        if self.path is not None:
            self.path.unlink(missing_ok=True)

    def add(self, entry: HistoryEntry) -> None:
        self.entries.append(entry)
        if self.path is None:
            return
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a", encoding="utf-8") as output:
            json.dump(entry.to_data(), output, ensure_ascii=False)
            output.write("\n")
            output.flush()

    def retrieve(
        self,
        query: str,
        *,
        exclude_text: str = "",
        limit: int = 3,
    ) -> list[HistoryEntry]:
        terms = _search_terms(query)
        lowered_query = query.casefold()
        ranked: list[tuple[int, int, HistoryEntry]] = []
        for index, entry in enumerate(self.entries):
            if entry.content and entry.content in exclude_text:
                continue
            haystack = " ".join(
                (entry.path, entry.name, entry.summary, entry.content)
            ).casefold()
            score = 0
            if entry.path and entry.path.casefold() in lowered_query:
                score += 12
            term_hits = sum(1 for term in terms if term in haystack)
            score += term_hits * 2
            if entry.kind == "step" and term_hits < 3:
                continue
            if score >= 4:
                ranked.append((score, index, entry))
        ranked.sort(key=lambda item: (item[0], item[1]), reverse=True)
        selected: list[HistoryEntry] = []
        seen_paths: set[str] = set()
        for _score, _index, entry in ranked:
            if entry.path:
                normalized = entry.path.casefold()
                if normalized in seen_paths:
                    continue
                seen_paths.add(normalized)
            selected.append(entry)
            if len(selected) >= limit:
                break
        return selected

    def retrieve_paths(
        self,
        paths: set[str],
        *,
        exclude_text: str = "",
        limit: int = 2,
    ) -> list[HistoryEntry]:
        normalized = {path.replace("\\", "/").casefold() for path in paths}
        selected: list[HistoryEntry] = []
        seen: set[str] = set()
        for entry in reversed(self.entries):
            path = entry.path.replace("\\", "/").casefold()
            if not path or path not in normalized or path in seen:
                continue
            if entry.summary.startswith("error:"):
                continue
            if entry.content and entry.content in exclude_text:
                continue
            selected.append(entry)
            seen.add(path)
            if len(selected) >= limit:
                break
        return selected

    def _load(self) -> None:
        if self.path is None or not self.path.exists():
            return
        for line in self.path.read_text(encoding="utf-8").splitlines():
            try:
                entry = HistoryEntry.from_data(json.loads(line))
            except json.JSONDecodeError:
                continue
            if entry is not None:
                self.entries.append(entry)


class TokenEstimator:
    """Estimate DeepSeek prompt tokens and calibrate against API usage."""

    def __init__(self) -> None:
        self._calibration = 1.0

    def text(self, value: str) -> int:
        ascii_characters = 0
        chinese_characters = 0
        other_characters = 0
        for character in value:
            codepoint = ord(character)
            if codepoint < 128:
                ascii_characters += 1
            elif 0x3400 <= codepoint <= 0x9FFF:
                chinese_characters += 1
            else:
                other_characters += 1
        raw = (
            ascii_characters * 0.3
            + chinese_characters * 0.6
            + other_characters * 0.8
        )
        return max(1, math.ceil(raw * self._calibration))

    def value(self, value: object) -> int:
        rendered = json.dumps(value, ensure_ascii=False, default=str)
        return self.text(rendered)

    def messages(self, messages: list[dict[str, object]]) -> int:
        return self.value(messages) + len(messages) * 4

    def calibrate(self, estimated_tokens: int, actual_tokens: int) -> None:
        if estimated_tokens <= 0 or actual_tokens <= 0:
            return
        observed = min(3.0, max(0.5, actual_tokens / estimated_tokens))
        self._calibration = self._calibration * 0.75 + observed * 0.25


class ContextManager:
    def __init__(
        self,
        *,
        max_prompt_tokens: int = DEFAULT_MAX_PROMPT_TOKENS,
        recent_turns: int = DEFAULT_RECENT_TURNS,
        enabled: bool = True,
        mode: str = "v3",
        archive_path: Path | None = None,
    ) -> None:
        if max_prompt_tokens < 800:
            raise ValueError("max_prompt_tokens must be at least 800")
        if recent_turns < 1:
            raise ValueError("recent_turns must be at least 1")
        if mode not in {"v2", "v3"}:
            raise ValueError("mode must be 'v2' or 'v3'")
        self.max_prompt_tokens = max_prompt_tokens
        self.recent_turns = recent_turns
        self.enabled = enabled
        self.mode = mode if enabled else "off"
        self.estimator = TokenEstimator()
        self.archive = ContextArchive(archive_path)
        self.last_stats = ContextStats()
        self._last_request_estimate = 0
        self._pending_retrieval_paths: set[str] = set()

    def start_task(self, *, reset: bool) -> None:
        self._pending_retrieval_paths.clear()
        if self.enabled and reset:
            self.archive.reset()

    def record_tool(
        self,
        *,
        step: int,
        name: str,
        arguments: Mapping[str, object],
        result: str,
        success: bool,
        version: str = "",
        file_content: str = "",
    ) -> None:
        if not self.enabled:
            return
        if name == "read_file" and result.startswith("unchanged read cache hit:"):
            path = arguments.get("path")
            if isinstance(path, str):
                self._pending_retrieval_paths.add(path)
            return
        path = arguments.get("path", arguments.get("cwd", ""))
        first_line = result.splitlines()[0] if result else "empty result"
        if success and not file_content and name == "write_file":
            candidate = arguments.get("content")
            if isinstance(candidate, str):
                file_content = candidate
        elif success and not file_content and name == "apply_patch":
            candidate = arguments.get("new_text")
            if isinstance(candidate, str):
                file_content = candidate
        outline = _file_outline(str(path), file_content) if file_content else ""
        summary = ("success" if success else "error") + f": {first_line}"
        if outline:
            summary += f"; {outline}"
        self.archive.add(
            HistoryEntry(
                step=step,
                kind="tool",
                name=name,
                path=str(path) if isinstance(path, str) else "",
                version=version,
                summary=summary,
                content=file_content or result,
            )
        )
        if not success and isinstance(path, str) and name in {
            "read_file",
            "write_file",
            "apply_patch",
        }:
            self._pending_retrieval_paths.add(path)
        if name == "run_command" and (not success or "Exit code: 0" not in result):
            self._pending_retrieval_paths.update(_mentioned_file_paths(result))

    def record_step(
        self,
        *,
        step: int,
        content: str | None,
        tools: list[str],
    ) -> None:
        if not self.enabled:
            return
        summary = "model selected " + (", ".join(tools) if tools else "final response")
        self.archive.add(
            HistoryEntry(
                step=step,
                kind="step",
                name="model_step",
                path="",
                version="",
                summary=summary,
                content=content or summary,
            )
        )

    def distill_tool_result(
        self,
        name: str,
        arguments: Mapping[str, object],
        result: str,
        *,
        success: bool,
    ) -> str:
        if not self.enabled:
            return result
        if not success or result.startswith("ERROR:"):
            return _truncate_to_tokens(result, self.estimator, 800)
        if name == "run_command":
            return _distill_command_result(result)
        if name in {"read_file", "search_text", "list_files"}:
            return _truncate_to_tokens(
                result,
                self.estimator,
                MAX_WORKING_RESULT_TOKENS,
            )
        if name in {"write_file", "apply_patch"}:
            path = arguments.get("path", "unknown")
            first_line = result.splitlines()[0] if result else "completed"
            return f"Modified file: {path}. {first_line}"
        return _truncate_to_tokens(result, self.estimator, 400)

    def build(
        self,
        messages: list[dict[str, object]],
        tail_messages: list[dict[str, object]] | None = None,
        *,
        contract_messages: list[dict[str, object]] | None = None,
        state_messages: list[dict[str, object]] | None = None,
        tools: list[dict[str, object]] | None = None,
    ) -> list[dict[str, object]]:
        # tail_messages is kept as a compatibility alias for state_messages.
        contract = contract_messages or []
        state = state_messages if state_messages is not None else (tail_messages or [])
        tool_definitions = tools or []

        anchor_end = _anchor_end(messages)
        anchors = [dict(message) for message in messages[:anchor_end]]
        history = messages[anchor_end:]
        full = [*anchors, *contract, *history, *state]
        tool_tokens = self.estimator.value(tool_definitions) if tool_definitions else 0
        original_tokens = self.estimator.messages(full) + tool_tokens

        if not self.enabled:
            self._set_stats(full, full, original_tokens, tool_tokens, 0)
            return full

        if self.mode == "v3":
            return self._build_v3(
                anchors=anchors,
                contract=contract,
                history=history,
                state=state,
                full=full,
                original_tokens=original_tokens,
                tool_tokens=tool_tokens,
            )

        turns = _split_turns(history)
        old_turns = turns[:-self.recent_turns]
        recent_turns = turns[-self.recent_turns:]
        recent = [message for turn in recent_turns for message in turn]
        recent_text = json.dumps(recent, ensure_ascii=False, default=str)
        query = json.dumps(
            [*anchors, *contract, *state, *recent],
            ensure_ascii=False,
            default=str,
        )
        retrieved = self.archive.retrieve(query, exclude_text=recent_text)
        retrieval = _retrieval_message(retrieved, self.estimator)
        retrieved_messages = [retrieval] if retrieval is not None else []
        candidate = [*anchors, *contract, *retrieved_messages, *recent, *state]

        while (
            self.estimator.messages(candidate) + tool_tokens > self.max_prompt_tokens
            and len(recent_turns) > 1
        ):
            old_turns.append(recent_turns.pop(0))
            recent = [message for turn in recent_turns for message in turn]
            candidate = [
                *anchors,
                *contract,
                *retrieved_messages,
                *recent,
                *state,
            ]

        if self.estimator.messages(candidate) + tool_tokens > self.max_prompt_tokens:
            compact_recent = _sanitize_turns(recent_turns, omit_tool_payload=True)
            compact_recent = _limit_tool_outputs(
                compact_recent,
                self.estimator,
                MAX_WORKING_RESULT_TOKENS,
            )
            candidate = [
                *anchors,
                *contract,
                *retrieved_messages,
                *compact_recent,
                *state,
            ]

        summarized = sum(len(turn) for turn in old_turns)
        self._set_stats(
            full,
            candidate,
            original_tokens,
            tool_tokens,
            summarized,
            retrieved_entries=len(retrieved),
        )
        return candidate

    def _build_v3(
        self,
        *,
        anchors: list[dict[str, object]],
        contract: list[dict[str, object]],
        history: list[dict[str, object]],
        state: list[dict[str, object]],
        full: list[dict[str, object]],
        original_tokens: int,
        tool_tokens: int,
    ) -> list[dict[str, object]]:
        turns = _split_turns(history)
        old_turns = turns[:-self.recent_turns]
        recent_turns = turns[-self.recent_turns:]
        recent = _sanitize_turns(recent_turns, omit_tool_payload=True)
        recent = _limit_tool_outputs(
            recent,
            self.estimator,
            MAX_WORKING_RESULT_TOKENS,
        )
        recent_text = json.dumps(recent, ensure_ascii=False, default=str)
        retrieved = self.archive.retrieve_paths(
            self._pending_retrieval_paths,
            exclude_text=recent_text,
        )
        self._pending_retrieval_paths.clear()
        retrieval = _retrieval_message(retrieved, self.estimator)
        retrieved_messages = [retrieval] if retrieval is not None else []
        ledger = _artifact_ledger_message(self.archive.entries)
        ledger_messages = [ledger] if ledger is not None else []
        candidate = [
            *anchors,
            *contract,
            *ledger_messages,
            *retrieved_messages,
            *recent,
            *state,
        ]

        while (
            self.estimator.messages(candidate) + tool_tokens > self.max_prompt_tokens
            and len(recent_turns) > 1
        ):
            old_turns.append(recent_turns.pop(0))
            recent = _sanitize_turns(recent_turns, omit_tool_payload=True)
            recent = _limit_tool_outputs(
                recent,
                self.estimator,
                MAX_WORKING_RESULT_TOKENS,
            )
            candidate = [
                *anchors,
                *contract,
                *ledger_messages,
                *retrieved_messages,
                *recent,
                *state,
            ]

        self._set_stats(
            full,
            candidate,
            original_tokens,
            tool_tokens,
            sum(len(turn) for turn in old_turns),
            retrieved_entries=len(retrieved),
        )
        return candidate

    def observe_prompt_usage(self, actual_prompt_tokens: int) -> None:
        self.estimator.calibrate(self._last_request_estimate, actual_prompt_tokens)
        self._last_request_estimate = 0

    def _set_stats(
        self,
        original: list[dict[str, object]],
        sent: list[dict[str, object]],
        original_tokens: int,
        tool_tokens: int,
        summarized_messages: int,
        retrieved_entries: int = 0,
    ) -> None:
        sent_tokens = self.estimator.messages(sent) + tool_tokens
        self._last_request_estimate = sent_tokens
        self.last_stats = ContextStats(
            original_characters=_message_size(original),
            sent_characters=_message_size(sent),
            original_tokens=original_tokens,
            sent_tokens=sent_tokens,
            tool_definition_tokens=tool_tokens,
            summarized_messages=summarized_messages,
            retrieved_entries=retrieved_entries,
        )


def _anchor_end(messages: list[dict[str, object]]) -> int:
    anchor_end = 0
    for message in messages:
        if message.get("role") not in {"system", "user"}:
            break
        anchor_end += 1
    return anchor_end


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


def _sanitize_turns(
    turns: list[list[dict[str, object]]],
    *,
    omit_tool_payload: bool,
) -> list[dict[str, object]]:
    return [
        _sanitize_message(message, omit_tool_payload=omit_tool_payload)
        for turn in turns
        for message in turn
    ]


def _sanitize_message(
    message: dict[str, object],
    *,
    omit_tool_payload: bool,
) -> dict[str, object]:
    sanitized = dict(message)
    tool_calls = message.get("tool_calls")
    if isinstance(tool_calls, list):
        sanitized["tool_calls"] = [
            _sanitize_tool_call(call, omit_payload=omit_tool_payload)
            if isinstance(call, dict)
            else call
            for call in tool_calls
        ]
    return sanitized


def _sanitize_tool_call(
    call: dict[str, object],
    *,
    omit_payload: bool,
) -> dict[str, object]:
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
        sanitized_function["arguments"] = arguments[:1_000]
    else:
        if isinstance(parsed, dict) and omit_payload:
            for key in ("content", "old_text", "new_text"):
                value = parsed.get(key)
                if isinstance(value, str) and len(value) > 300:
                    parsed[key] = (
                        f"<omitted {len(value)} characters; current file is in workspace>"
                    )
        sanitized_function["arguments"] = json.dumps(parsed, ensure_ascii=False)
    sanitized["function"] = sanitized_function
    return sanitized


def _limit_tool_outputs(
    messages: list[dict[str, object]],
    estimator: TokenEstimator,
    token_limit: int,
) -> list[dict[str, object]]:
    limited: list[dict[str, object]] = []
    for message in messages:
        copy = dict(message)
        if message.get("role") == "tool":
            content = str(message.get("content", ""))
            copy["content"] = _truncate_to_tokens(content, estimator, token_limit)
        limited.append(copy)
    return limited


def _truncate_to_tokens(
    text: str,
    estimator: TokenEstimator,
    token_limit: int,
) -> str:
    if estimator.text(text) <= token_limit:
        return text
    low = 0
    high = len(text)
    while low < high:
        middle = (low + high + 1) // 2
        if estimator.text(text[:middle]) <= token_limit:
            low = middle
        else:
            high = middle - 1
    return f"{text[:low]}\n... tool output compacted; full result is in local history ..."


def _retrieval_message(
    entries: list[HistoryEntry],
    estimator: TokenEstimator,
) -> dict[str, object] | None:
    if not entries:
        return None
    lines = ["<retrieved_history>"]
    for entry in entries:
        identity = f"step={entry.step} {entry.name}"
        if entry.path:
            identity += f" path={entry.path}"
        if entry.version:
            identity += f" version={entry.version}"
        snippet = _truncate_to_tokens(entry.content, estimator, 400)
        lines.extend((f"- {identity}: {entry.summary}", snippet))
    lines.append("</retrieved_history>")
    return {"role": "system", "content": "\n".join(lines)}


def _artifact_ledger_message(
    entries: list[HistoryEntry],
) -> dict[str, object] | None:
    latest: dict[str, HistoryEntry] = {}
    for entry in reversed(entries):
        if entry.name not in {"write_file", "apply_patch"} or not entry.path:
            continue
        if entry.summary.startswith("error:"):
            continue
        normalized = entry.path.replace("\\", "/")
        if normalized.casefold() not in {path.casefold() for path in latest}:
            latest[normalized] = entry
        if len(latest) >= 16:
            break
    if not latest:
        return None
    lines = [
        "<artifact_ledger>",
        "Files below are already saved. Do not read them again merely to confirm "
        "their contents; use tests or read only a specific file needed for repair.",
    ]
    for path, entry in reversed(list(latest.items())):
        version = f" version={entry.version[:12]}" if entry.version else ""
        lines.append(f"- {path}{version}: {entry.summary}")
    lines.append("</artifact_ledger>")
    return {"role": "system", "content": "\n".join(lines)}


def _file_outline(path: str, content: str) -> str:
    if not content:
        return ""
    size = len(content)
    if not path.casefold().endswith(".py"):
        return f"{size} characters"
    try:
        tree = ast.parse(content)
    except SyntaxError:
        return f"{size} characters; Python syntax not yet valid"
    symbols: list[str] = []
    for node in tree.body:
        if isinstance(node, ast.ClassDef):
            methods = [
                child.name
                for child in node.body
                if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef))
            ]
            detail = f"class {node.name}"
            if methods:
                detail += f"({', '.join(methods[:8])})"
            symbols.append(detail)
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            symbols.append(f"def {node.name}")
    rendered = ", ".join(symbols[:12]) or "no top-level classes/functions"
    return f"{size} characters; symbols: {rendered}"


def _mentioned_file_paths(text: str) -> set[str]:
    return {
        match.replace("\\", "/")
        for match in re.findall(
            r"(?<![\w.-])(?:[\w.-]+[\\/])*[\w.-]+\.[A-Za-z0-9]{1,8}",
            text,
        )
    }


def _search_terms(text: str) -> set[str]:
    matches = re.findall(
        r"[A-Za-z_][A-Za-z0-9_./\\-]{2,}|[\u3400-\u9fff]{2,}",
        text.casefold(),
    )
    ignored = {"system", "content", "role", "current", "state", "task"}
    return {term for term in matches if term not in ignored}


def _distill_command_result(result: str) -> str:
    lines = result.splitlines()
    if not lines:
        return "Command returned no output"
    selected = [lines[0]]
    evidence_patterns = (
        r"^Ran \d+ tests?",
        r"^OK(?: \(|$)",
        r"^FAILED(?: \(|$)",
        r"^ERROR(?:\b|:)",
        r"^FAIL(?:\b|:)",
        r"Traceback",
        r"(?:Assertion|Import|ModuleNotFound|Syntax|Type|Value)Error",
        r"\bpassed\b|\bfailed\b|\berrors?=|\bfailures?=",
    )
    for line in lines[1:]:
        if any(re.search(pattern, line, re.IGNORECASE) for pattern in evidence_patterns):
            if line not in selected:
                selected.append(line)
    if len(selected) == 1 and len(lines) > 1:
        selected.extend(lines[-10:])
    return "\n".join(selected[:40])


def _message_size(messages: list[dict[str, object]]) -> int:
    return len(json.dumps(messages, ensure_ascii=False, default=str))
