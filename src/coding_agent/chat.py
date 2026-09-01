"""Minimal local chat UI for the coding agent."""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from threading import RLock, Thread
from typing import Callable, Sequence
from urllib.parse import urlparse
from uuid import uuid4

from coding_agent.agent import Agent, AgentError
from coding_agent.checkpoint import CheckpointStore
from coding_agent.config import ConfigError, Settings
from coding_agent.model import DeepSeekModel, ModelError
from coding_agent.project_memory import ProjectMemoryStore, snapshot_structure
from coding_agent.trace import JsonlTrace
from coding_agent.workspace import Workspace, WorkspaceError


@dataclass
class ChatMessage:
    role: str
    content: str
    events: list[str] = field(default_factory=list)
    created_at: str = ""
    id: str = ""
    parent_id: str = ""

    def to_data(self) -> dict[str, object]:
        return {
            "id": self.id,
            "parent_id": self.parent_id,
            "role": self.role,
            "content": self.content,
            "events": self.events,
            "created_at": self.created_at,
        }

    @classmethod
    def from_data(cls, data: object) -> "ChatMessage | None":
        if not isinstance(data, dict):
            return None
        events = data.get("events", [])
        if not isinstance(events, list):
            events = []
        return cls(
            role=str(data.get("role", "")),
            content=str(data.get("content", "")),
            events=[str(event) for event in events],
            created_at=str(data.get("created_at", "")),
            id=str(data.get("id", "")),
            parent_id=str(data.get("parent_id", "")),
        )


@dataclass
class RunningConversation:
    started_at: str
    user_message_id: str = ""
    events: list[str] = field(default_factory=list)

    def to_data(self) -> dict[str, object]:
        return {
            "started_at": self.started_at,
            "user_message_id": self.user_message_id,
            "events": list(self.events),
        }


@dataclass
class Conversation:
    id: str
    project_id: str
    title: str
    messages: list[ChatMessage] = field(default_factory=list)
    created_at: str = ""
    updated_at: str = ""
    pinned: bool = False
    current_message_id: str = ""

    def to_data(self) -> dict[str, object]:
        return {
            "id": self.id,
            "project_id": self.project_id,
            "title": self.title,
            "messages": [message.to_data() for message in self.messages],
            "current_message_id": self.current_message_id,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "pinned": self.pinned,
        }

    def current_messages(self) -> list[ChatMessage]:
        if not self.messages:
            return []
        by_id = {message.id: message for message in self.messages if message.id}
        current_id = self.current_message_id
        if current_id not in by_id:
            current_id = self.messages[-1].id
        path: list[ChatMessage] = []
        seen: set[str] = set()
        while current_id and current_id in by_id and current_id not in seen:
            seen.add(current_id)
            message = by_id[current_id]
            path.append(message)
            current_id = message.parent_id
        return list(reversed(path))

    def contains_message(self, message_id: str) -> bool:
        return any(message.id == message_id for message in self.messages)

    @classmethod
    def from_data(cls, data: object) -> "Conversation | None":
        if not isinstance(data, dict):
            return None
        messages = [
            message
            for item in data.get("messages", [])
            if (message := ChatMessage.from_data(item)) is not None
        ]
        current_message_id = _ensure_message_tree(
            messages,
            str(data.get("current_message_id", "")),
        )
        return cls(
            id=str(data.get("id", "")),
            project_id=str(data.get("project_id", "")),
            title=str(data.get("title", "未命名对话")),
            messages=messages,
            created_at=str(data.get("created_at", "")),
            updated_at=str(data.get("updated_at", "")),
            pinned=bool(data.get("pinned", False)),
            current_message_id=current_message_id,
        )


def _ensure_message_tree(
    messages: list[ChatMessage],
    current_message_id: str,
) -> str:
    seen: set[str] = set()
    previous_id = ""
    for message in messages:
        if not message.id or message.id in seen:
            message.id = uuid4().hex
        seen.add(message.id)
        if not message.parent_id and previous_id:
            message.parent_id = previous_id
        previous_id = message.id
    if current_message_id and current_message_id in seen:
        return current_message_id
    return messages[-1].id if messages else ""


class ChatStore:
    def __init__(self, path: Path) -> None:
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.conversations: dict[str, Conversation] = {}
        self._load()

    def list_conversations(self) -> list[Conversation]:
        return sorted(
            self.conversations.values(),
            key=lambda item: (item.pinned, item.updated_at or item.created_at),
            reverse=True,
        )

    def create_conversation(self, project_id: str, title: str = "新对话") -> Conversation:
        now = _now()
        conversation = Conversation(
            id=uuid4().hex,
            project_id=project_id,
            title=title.strip() or "新对话",
            created_at=now,
            updated_at=now,
        )
        self.conversations[conversation.id] = conversation
        self.save()
        return conversation

    def get(self, conversation_id: str) -> Conversation:
        try:
            return self.conversations[conversation_id]
        except KeyError as error:
            raise KeyError(f"Conversation not found: {conversation_id}") from error

    def rename_conversation(self, conversation_id: str, title: str) -> Conversation:
        name = title.strip()
        if not name:
            raise ValueError("Conversation title cannot be empty")
        conversation = self.get(conversation_id)
        conversation.title = name
        conversation.updated_at = _now()
        self.save()
        return conversation

    def pin_conversation(self, conversation_id: str, pinned: bool) -> Conversation:
        conversation = self.get(conversation_id)
        conversation.pinned = pinned
        conversation.updated_at = _now()
        self.save()
        return conversation

    def delete_conversation(self, conversation_id: str) -> None:
        self.get(conversation_id)
        del self.conversations[conversation_id]
        self.save()

    def checkout_message(self, conversation_id: str, message_id: str) -> Conversation:
        conversation = self.get(conversation_id)
        if message_id and not conversation.contains_message(message_id):
            raise KeyError(f"Message not found: {message_id}")
        conversation.current_message_id = message_id
        conversation.updated_at = _now()
        self.save()
        return conversation

    def append_user_message(
        self,
        conversation_id: str,
        content: str,
        *,
        parent_id: str | None = None,
    ) -> ChatMessage:
        conversation = self.get(conversation_id)
        resolved_parent_id = conversation.current_message_id if parent_id is None else parent_id
        if resolved_parent_id and not conversation.contains_message(resolved_parent_id):
            raise KeyError(f"Message not found: {resolved_parent_id}")
        now = _now()
        message = ChatMessage(
            "user",
            content,
            created_at=now,
            id=uuid4().hex,
            parent_id=resolved_parent_id or "",
        )
        conversation.messages.append(message)
        conversation.current_message_id = message.id
        if conversation.title == "新对话":
            conversation.title = content.strip()[:32] or "新对话"
        conversation.updated_at = now
        self.save()
        return message

    def append_assistant_message(
        self,
        conversation_id: str,
        *,
        content: str,
        events: list[str],
        parent_id: str | None = None,
    ) -> ChatMessage:
        conversation = self.get(conversation_id)
        resolved_parent_id = conversation.current_message_id if parent_id is None else parent_id
        if resolved_parent_id and not conversation.contains_message(resolved_parent_id):
            raise KeyError(f"Message not found: {resolved_parent_id}")
        now = _now()
        message = ChatMessage(
            "assistant",
            content,
            events=events,
            created_at=now,
            id=uuid4().hex,
            parent_id=resolved_parent_id or "",
        )
        conversation.messages.append(message)
        conversation.current_message_id = message.id
        conversation.updated_at = now
        self.save()
        return message

    def append_pair(
        self,
        conversation_id: str,
        *,
        user_message: str,
        assistant_message: str,
        events: list[str],
    ) -> Conversation:
        user = self.append_user_message(conversation_id, user_message)
        self.append_assistant_message(
            conversation_id,
            content=assistant_message,
            events=events,
            parent_id=user.id,
        )
        return self.get(conversation_id)

    def save(self) -> None:
        data = {
            "version": 1,
            "conversations": [
                conversation.to_data()
                for conversation in self.list_conversations()
            ],
        }
        temporary = self.path.with_suffix(".tmp")
        temporary.write_text(
            json.dumps(data, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        temporary.replace(self.path)

    def _load(self) -> None:
        if not self.path.exists():
            return
        try:
            data = json.loads(self.path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return
        if not isinstance(data, dict) or data.get("version") != 1:
            return
        for item in data.get("conversations", []):
            conversation = Conversation.from_data(item)
            if conversation is not None and conversation.id:
                self.conversations[conversation.id] = conversation


class ChatApp:
    def __init__(
        self,
        data_dir: Path,
        settings: Settings,
        *,
        directory_picker: Callable[[], str] | None = None,
    ) -> None:
        self.data_dir = data_dir
        self.settings = settings
        self.memory_store = ProjectMemoryStore(data_dir / "project-memory")
        self.chat_store = ChatStore(data_dir / "chat-ui" / "conversations.json")
        self._agents: dict[str, Agent] = {}
        self._running_conversations: dict[str, RunningConversation] = {}
        self._lock = RLock()
        self.directory_picker = directory_picker or _choose_directory

    def state(self) -> dict[str, object]:
        with self._lock:
            projects = [
                {
                    "id": memory.project_id,
                    "workspace": memory.workspace,
                    "display_name": memory.display_name,
                    "updated_at": memory.updated_at,
                }
                for memory in self.memory_store.list_projects()
            ]
            conversations = [
                self._conversation_payload(conversation)
                for conversation in self.chat_store.list_conversations()
            ]
            running = {
                conversation_id: item.to_data()
                for conversation_id, item in self._running_conversations.items()
            }
        return {
            "projects": projects,
            "conversations": conversations,
            "running_conversations": running,
        }

    def add_project(self, path: str) -> dict[str, object]:
        workspace = Workspace(Path(path))
        memory = self.memory_store.load(workspace)
        memory.workspace = str(workspace.root)
        if not memory.display_name:
            memory.display_name = workspace.root.name or str(workspace.root)
        memory.structure = snapshot_structure(workspace)
        memory.updated_at = _now()
        self.memory_store.save(memory)
        return {
            "id": memory.project_id,
            "workspace": memory.workspace,
            "display_name": memory.display_name,
            "updated_at": memory.updated_at,
        }

    def pick_project(self) -> dict[str, object]:
        path = self.directory_picker().strip()
        if not path:
            raise ValueError("No directory selected")
        return self.add_project(path)

    def rename_project(self, project_id: str, display_name: str) -> dict[str, object]:
        memory = self.memory_store.rename_project(project_id, display_name)
        return {
            "id": memory.project_id,
            "workspace": memory.workspace,
            "display_name": memory.display_name,
            "updated_at": memory.updated_at,
        }

    def create_conversation(self, project_id: str, title: str = "新对话") -> dict[str, object]:
        self._workspace_for_project(project_id)
        conversation = self.chat_store.create_conversation(project_id, title)
        return self._conversation_payload(conversation)

    def rename_conversation(self, conversation_id: str, title: str) -> dict[str, object]:
        conversation = self.chat_store.rename_conversation(conversation_id, title)
        return self._conversation_payload(conversation)

    def pin_conversation(self, conversation_id: str, pinned: bool) -> dict[str, object]:
        conversation = self.chat_store.pin_conversation(conversation_id, pinned)
        return self._conversation_payload(conversation)

    def checkout_message(
        self,
        conversation_id: str,
        message_id: str,
    ) -> dict[str, object]:
        with self._lock:
            if conversation_id in self._running_conversations:
                raise RuntimeError("Cannot switch tree nodes while a conversation is running")
            conversation = self.chat_store.checkout_message(conversation_id, message_id)
            self._agents.pop(conversation_id, None)
            return self._conversation_payload(conversation)

    def delete_conversation(self, conversation_id: str) -> dict[str, object]:
        with self._lock:
            if conversation_id in self._running_conversations:
                raise RuntimeError("Cannot delete a running conversation")
            self.chat_store.delete_conversation(conversation_id)
            self._agents.pop(conversation_id, None)
        return {"deleted": conversation_id}

    def send_message(self, conversation_id: str, content: str) -> dict[str, object]:
        conversation = self.chat_store.get(conversation_id)
        workspace_path = self._workspace_for_project(conversation.project_id)
        workspace = Workspace(Path(workspace_path))
        events: list[str] = []
        user_message = self.chat_store.append_user_message(conversation.id, content)
        self._running_conversations[conversation.id] = RunningConversation(
            started_at=_now(),
            user_message_id=user_message.id,
        )

        def on_event(event: str) -> None:
            events.append(event)
            running = self._running_conversations.get(conversation.id)
            if running is not None:
                running.events.append(event)

        agent = self._agents.get(conversation.id)
        if agent is None:
            agent = self._create_agent(workspace, on_event)
            self._agents[conversation.id] = agent
        else:
            agent.on_event = on_event
        try:
            answer = agent.run(content)
        except (AgentError, ModelError, OSError, WorkspaceError) as error:
            answer = f"任务失败: {error}"
        finally:
            self._running_conversations.pop(conversation.id, None)
        self.chat_store.append_assistant_message(
            conversation.id,
            content=answer,
            events=events,
            parent_id=user_message.id,
        )
        saved = self.chat_store.get(conversation.id)
        return {
            "conversation": self._conversation_payload(saved),
            "answer": answer,
            "events": events,
        }

    def start_message(
        self,
        conversation_id: str,
        content: str,
        parent_message_id: str = "",
    ) -> dict[str, object]:
        message = content.strip()
        if not message:
            raise ValueError("Message cannot be empty")
        with self._lock:
            if conversation_id in self._running_conversations:
                raise RuntimeError("Conversation is already running")
            conversation = self.chat_store.get(conversation_id)
            workspace_path = self._workspace_for_project(conversation.project_id)
            user_message = self.chat_store.append_user_message(
                conversation.id,
                message,
                parent_id=parent_message_id or None,
            )
            saved = self.chat_store.get(conversation.id)
            running = RunningConversation(
                started_at=_now(),
                user_message_id=user_message.id,
            )
            self._running_conversations[conversation.id] = running
            payload = {
                "conversation": self._conversation_payload(saved),
                "running": running.to_data(),
            }
        thread = Thread(
            target=self._run_message_worker,
            args=(conversation_id, workspace_path, message, user_message.id),
            daemon=True,
        )
        thread.start()
        return payload

    def _run_message_worker(
        self,
        conversation_id: str,
        workspace_path: str,
        content: str,
        user_message_id: str,
    ) -> None:
        def record_event(event: str) -> None:
            with self._lock:
                running = self._running_conversations.get(conversation_id)
                if running is not None:
                    running.events.append(event)

        events: list[str] = []

        def on_event(event: str) -> None:
            events.append(event)
            record_event(event)

        try:
            workspace = Workspace(Path(workspace_path))
            with self._lock:
                agent = self._agents.get(conversation_id)
                if agent is None:
                    agent = self._create_agent(workspace, on_event)
                    self._agents[conversation_id] = agent
                else:
                    agent.on_event = on_event
            answer = agent.run(content)
        except (AgentError, ModelError, OSError, WorkspaceError) as error:
            answer = f"任务失败: {error}"
        except Exception as error:  # Keep the UI from hanging on unexpected failures.
            answer = f"任务失败: {error}"
        finally:
            with self._lock:
                self.chat_store.append_assistant_message(
                    conversation_id,
                    content=answer,
                    events=events,
                    parent_id=user_message_id,
                )
                self._running_conversations.pop(conversation_id, None)

    def _create_agent(
        self,
        workspace: Workspace,
        on_event: Callable[[str], None],
    ) -> Agent:
        trace = JsonlTrace.create(self.data_dir / "traces")
        checkpoint = CheckpointStore.for_workspace(
            self.data_dir / "checkpoints",
            workspace.root,
        )
        return Agent(
            DeepSeekModel(self.settings),
            workspace,
            on_event=on_event,
            trace=trace,
            checkpoint_store=checkpoint,
            memory_store=self.memory_store,
        )

    def _workspace_for_project(self, project_id: str) -> str:
        for memory in self.memory_store.list_projects():
            if memory.project_id == project_id:
                return memory.workspace
        raise KeyError(f"Project not found: {project_id}")

    @staticmethod
    def _conversation_payload(conversation: Conversation) -> dict[str, object]:
        return {
            "id": conversation.id,
            "project_id": conversation.project_id,
            "title": conversation.title,
            "messages": [
                message.to_data()
                for message in conversation.current_messages()
            ],
            "message_tree": [
                message.to_data()
                for message in conversation.messages
            ],
            "current_message_id": conversation.current_message_id,
            "updated_at": conversation.updated_at,
            "pinned": conversation.pinned,
        }


def make_handler(app: ChatApp) -> type[BaseHTTPRequestHandler]:
    class Handler(BaseHTTPRequestHandler):
        def do_GET(self) -> None:
            parsed = urlparse(self.path)
            if parsed.path == "/":
                self._send_html(INDEX_HTML)
                return
            if parsed.path == "/api/state":
                self._send_json(app.state())
                return
            self.send_error(HTTPStatus.NOT_FOUND)

        def do_POST(self) -> None:
            parsed = urlparse(self.path)
            try:
                payload = self._read_json()
                if parsed.path == "/api/projects":
                    result = app.add_project(str(payload.get("path", "")))
                elif parsed.path == "/api/projects/pick":
                    result = app.pick_project()
                elif parsed.path == "/api/projects/rename":
                    result = app.rename_project(
                        str(payload.get("project_id", "")),
                        str(payload.get("display_name", "")),
                    )
                elif parsed.path == "/api/conversations":
                    result = app.create_conversation(
                        str(payload.get("project_id", "")),
                        str(payload.get("title", "新对话")),
                    )
                elif parsed.path == "/api/conversations/rename":
                    result = app.rename_conversation(
                        str(payload.get("conversation_id", "")),
                        str(payload.get("title", "")),
                    )
                elif parsed.path == "/api/conversations/pin":
                    result = app.pin_conversation(
                        str(payload.get("conversation_id", "")),
                        bool(payload.get("pinned", False)),
                    )
                elif parsed.path == "/api/conversations/delete":
                    result = app.delete_conversation(
                        str(payload.get("conversation_id", "")),
                    )
                elif parsed.path == "/api/conversations/checkout":
                    result = app.checkout_message(
                        str(payload.get("conversation_id", "")),
                        str(payload.get("message_id", "")),
                    )
                elif parsed.path == "/api/messages":
                    result = app.start_message(
                        str(payload.get("conversation_id", "")),
                        str(payload.get("content", "")),
                        str(payload.get("parent_message_id", "")),
                    )
                else:
                    self.send_error(HTTPStatus.NOT_FOUND)
                    return
            except (KeyError, RuntimeError, ValueError, WorkspaceError) as error:
                self._send_json({"error": str(error)}, status=HTTPStatus.BAD_REQUEST)
                return
            self._send_json(result)

        def log_message(self, format: str, *args: object) -> None:
            return None

        def _read_json(self) -> dict[str, object]:
            length = int(self.headers.get("Content-Length", "0"))
            raw = self.rfile.read(length).decode("utf-8") if length else "{}"
            data = json.loads(raw)
            if not isinstance(data, dict):
                raise ValueError("Request body must be a JSON object")
            return data

        def _send_json(
            self,
            data: dict[str, object],
            *,
            status: HTTPStatus = HTTPStatus.OK,
        ) -> None:
            body = json.dumps(data, ensure_ascii=False).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def _send_html(self, html: str) -> None:
            body = html.encode("utf-8")
            self.send_response(HTTPStatus.OK)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

    return Handler


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the minimal Coding Agent chat UI")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path.cwd() / ".coding-agent",
        help="Directory for traces, checkpoints, project memory, and chat history",
    )
    return parser


def _choose_directory() -> str:
    try:
        import tkinter as tk
        from tkinter import filedialog
    except ImportError as error:
        raise RuntimeError("Directory picker is not available") from error

    root = tk.Tk()
    root.withdraw()
    root.attributes("-topmost", True)
    try:
        selected = filedialog.askdirectory(title="选择项目文件夹")
    finally:
        root.destroy()
    return selected


def main(argv: Sequence[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    try:
        settings = Settings.load(Path.cwd())
    except ConfigError as error:
        print(f"配置错误: {error}")
        return 2
    app = ChatApp(args.data_dir, settings)
    server = ThreadingHTTPServer((args.host, args.port), make_handler(app))
    print(f"聊天界面: http://{args.host}:{args.port}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print()
        return 0
    finally:
        server.server_close()


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


INDEX_HTML = r"""<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Coding Agent Chat</title>
  <style>
    :root {
      color-scheme: light;
      --bg: #f6f7f9;
      --panel: #ffffff;
      --line: #d7dce2;
      --text: #1f2933;
      --muted: #6b7280;
      --accent: #0f766e;
      --accent-strong: #0b5f59;
      --user: #e8f5f3;
      --assistant: #ffffff;
      font-family: "Segoe UI", system-ui, sans-serif;
    }
    * { box-sizing: border-box; }
    body {
      margin: 0;
      background: var(--bg);
      color: var(--text);
      height: 100vh;
      overflow: hidden;
    }
    .app {
      display: grid;
      grid-template-columns: 320px minmax(0, 1fr);
      height: 100vh;
    }
    aside {
      border-right: 1px solid var(--line);
      background: var(--panel);
      display: grid;
      grid-template-rows: auto 1fr;
      min-width: 0;
    }
    .sidebar-top {
      padding: 16px;
      border-bottom: 1px solid var(--line);
    }
    h1 {
      font-size: 18px;
      margin: 0 0 14px;
      font-weight: 650;
    }
    label {
      display: block;
      font-size: 12px;
      color: var(--muted);
      margin-bottom: 6px;
    }
    input, select, textarea, button {
      font: inherit;
    }
    input, select, textarea {
      width: 100%;
      border: 1px solid var(--line);
      border-radius: 6px;
      padding: 9px 10px;
      background: #fff;
      color: var(--text);
    }
    button {
      border: 1px solid var(--accent);
      border-radius: 6px;
      background: var(--accent);
      color: #fff;
      padding: 9px 12px;
      cursor: pointer;
      white-space: nowrap;
    }
    button.secondary {
      background: #fff;
      color: var(--accent-strong);
    }
    button:disabled {
      opacity: .55;
      cursor: not-allowed;
    }
    .row {
      display: flex;
      gap: 8px;
      align-items: center;
      margin-bottom: 12px;
    }
    .row input, .row select { min-width: 0; }
    .project-actions {
      display: grid;
      grid-template-columns: minmax(0, 1fr) auto;
      gap: 8px;
      margin-bottom: 12px;
    }
    .current-project {
      min-width: 0;
      border: 1px solid var(--line);
      border-radius: 6px;
      padding: 9px 10px;
      color: var(--muted);
      overflow: hidden;
      text-overflow: ellipsis;
      white-space: nowrap;
      background: #fff;
    }
    .lists {
      min-height: 0;
      overflow: auto;
      padding: 12px;
    }
    .section-title {
      font-size: 12px;
      color: var(--muted);
      margin: 10px 4px 6px;
      text-transform: uppercase;
      letter-spacing: .04em;
    }
    .project-group {
      margin-bottom: 8px;
    }
    .project-row {
      display: grid;
      grid-template-columns: auto minmax(0, 1fr) auto auto;
      gap: 6px;
      align-items: center;
      width: 100%;
      border: 1px solid transparent;
      border-radius: 6px;
      background: transparent;
      color: var(--text);
      padding: 8px 6px;
    }
    .project-row.active {
      background: #edf7f5;
      border-color: #b8dcd7;
    }
    .project-toggle,
    .project-name {
      border: 0;
      background: transparent;
      color: inherit;
      padding: 0;
      text-align: left;
    }
    .project-toggle {
      width: 20px;
      text-align: center;
    }
    .project-name {
      overflow: hidden;
      text-overflow: ellipsis;
    }
    .project-action,
    .conversation-action {
      width: 30px;
      height: 30px;
      border-color: transparent;
      background: transparent;
      color: var(--muted);
      padding: 0;
      text-align: center;
    }
    .project-action:hover,
    .conversation-action:hover {
      border-color: var(--line);
      color: var(--accent-strong);
      background: #fff;
    }
    .conversation-action.danger:hover {
      color: #b42318;
    }
    .conversation-list {
      margin: 2px 0 10px 24px;
    }
    .empty-project {
      color: var(--muted);
      font-size: 13px;
      padding: 6px 8px;
    }
    .item-row {
      display: grid;
      grid-template-columns: minmax(0, 1fr) auto auto auto;
      gap: 6px;
      align-items: center;
      margin-bottom: 4px;
    }
    .item {
      width: 100%;
      border: 1px solid transparent;
      border-radius: 6px;
      background: transparent;
      color: var(--text);
      text-align: left;
      padding: 9px 10px;
      overflow: hidden;
      text-overflow: ellipsis;
    }
    .item.pinned::before {
      content: "置顶 ";
      color: var(--accent-strong);
      font-size: 12px;
    }
    .item.active {
      background: #edf7f5;
      border-color: #b8dcd7;
    }
    main {
      display: grid;
      grid-template-rows: auto 1fr auto;
      min-width: 0;
      height: 100vh;
    }
    header {
      background: var(--panel);
      border-bottom: 1px solid var(--line);
      padding: 15px 20px;
    }
    .title {
      font-weight: 650;
      overflow: hidden;
      text-overflow: ellipsis;
      white-space: nowrap;
    }
    .subtitle {
      color: var(--muted);
      font-size: 13px;
      margin-top: 4px;
      overflow: hidden;
      text-overflow: ellipsis;
      white-space: nowrap;
    }
    .messages {
      min-height: 0;
      overflow: auto;
      padding: 20px;
    }
    .conversation-body {
      display: grid;
      grid-template-columns: 180px minmax(0, 1fr);
      min-height: 0;
    }
    .day-nav {
      border-right: 1px solid var(--line);
      background: #fbfcfc;
      overflow: auto;
      padding: 18px 8px;
    }
    .day-link {
      width: 100%;
      border: 0;
      border-left: 2px solid var(--line);
      border-radius: 0;
      background: transparent;
      color: var(--muted);
      padding: 7px 0 7px 8px;
      text-align: left;
      font-size: 12px;
    }
    .day-link:hover {
      color: var(--accent-strong);
      border-left-color: var(--accent);
    }
    .day-empty {
      color: var(--muted);
      font-size: 12px;
      padding: 8px 0;
      text-align: center;
    }
    .nav-group-title {
      color: var(--muted);
      font-size: 11px;
      margin: 10px 0 6px;
      text-transform: uppercase;
      letter-spacing: .04em;
    }
    .session-tree {
      display: grid;
      gap: 3px;
    }
    .tree-node {
      width: 100%;
      border: 0;
      border-left: 2px solid var(--line);
      border-radius: 0;
      background: transparent;
      color: var(--muted);
      padding: 6px 4px 6px 8px;
      text-align: left;
      font-size: 12px;
      line-height: 1.3;
    }
    .tree-node.on-path {
      color: var(--text);
      border-left-color: #c7d7d4;
    }
    .tree-node.active {
      color: var(--accent-strong);
      border-left-color: var(--accent);
      background: #edf7f5;
    }
    .tree-label {
      display: block;
      overflow: hidden;
      text-overflow: ellipsis;
      white-space: nowrap;
    }
    .branch-count {
      color: var(--accent-strong);
      font-size: 11px;
    }
    .day-separator {
      max-width: 900px;
      margin: 6px 0 14px;
      color: var(--muted);
      font-size: 12px;
      border-bottom: 1px solid var(--line);
      padding-bottom: 8px;
    }
    .message {
      max-width: 900px;
      margin: 0 0 14px;
      border: 1px solid var(--line);
      border-radius: 8px;
      padding: 12px 14px;
      background: var(--assistant);
      line-height: 1.5;
    }
    .message-content {
      overflow-wrap: anywhere;
    }
    .message-content > :first-child { margin-top: 0; }
    .message-content > :last-child { margin-bottom: 0; }
    .message-content h2 {
      font-size: 18px;
      line-height: 1.35;
      margin: 18px 0 8px;
    }
    .message-content h3 {
      font-size: 15px;
      line-height: 1.35;
      margin: 14px 0 6px;
    }
    .message-content p {
      margin: 0 0 10px;
    }
    .message-content ul,
    .message-content ol {
      margin: 0 0 12px 20px;
      padding: 0;
    }
    .message-content li {
      margin: 4px 0;
    }
    .message-content code {
      border: 1px solid var(--line);
      border-radius: 4px;
      background: #f7faf9;
      padding: 1px 4px;
      font-family: Consolas, "SFMono-Regular", monospace;
      font-size: .92em;
    }
    .message-content pre {
      margin: 10px 0 12px;
      overflow: auto;
      border: 1px solid var(--line);
      border-radius: 6px;
      background: #f7faf9;
      padding: 10px 12px;
    }
    .message-content pre code {
      border: 0;
      border-radius: 0;
      background: transparent;
      padding: 0;
      white-space: pre;
    }
    .message.user {
      margin-left: auto;
      background: var(--user);
      border-color: #c7e7e2;
    }
    .message.pending {
      color: var(--muted);
      border-color: transparent;
      background: transparent;
      padding-left: 0;
    }
    .elapsed {
      color: var(--muted);
      font-size: 13px;
      margin-bottom: 12px;
      border-bottom: 1px solid var(--line);
      padding-bottom: 12px;
    }
    .events {
      margin-top: 10px;
      border-top: 1px solid var(--line);
      padding-top: 8px;
      color: var(--muted);
      font-size: 13px;
      white-space: pre-wrap;
    }
    .events-details {
      margin-top: 10px;
      border-top: 1px solid var(--line);
      padding-top: 8px;
      color: var(--muted);
      font-size: 13px;
    }
    .events-details summary {
      cursor: pointer;
      user-select: none;
    }
    .events-details .events {
      border-top: 0;
      margin-top: 8px;
      padding-top: 0;
    }
    .composer {
      background: var(--panel);
      border-top: 1px solid var(--line);
      padding: 14px 20px;
      display: grid;
      grid-template-columns: minmax(0, 1fr) auto;
      gap: 10px;
    }
    textarea {
      min-height: 42px;
      height: 42px;
      max-height: 160px;
      overflow-y: hidden;
      resize: vertical;
      transition: height .12s ease;
    }
    textarea.expanded {
      min-height: 84px;
    }
    @media (max-width: 820px) {
      .app { grid-template-columns: 1fr; }
      aside {
        height: 42vh;
        border-right: 0;
        border-bottom: 1px solid var(--line);
      }
      body { overflow: auto; }
      main { height: 58vh; }
      .conversation-body { grid-template-columns: 1fr; }
      .day-nav { display: none; }
    }
  </style>
</head>
<body>
  <div class="app">
    <aside>
      <div class="sidebar-top">
        <h1>Coding Agent</h1>
        <label for="projectPath">项目路径</label>
        <div class="row">
          <input id="projectPath" placeholder="D:\path\to\project">
          <button id="browseProject" class="secondary">浏览</button>
          <button id="addProject">添加</button>
        </div>
        <label>当前项目</label>
        <div class="project-actions">
          <div class="current-project" id="currentProjectLabel">未选择项目</div>
          <button id="newConversation" class="secondary">新对话</button>
        </div>
      </div>
      <div class="lists">
        <div class="section-title">项目与对话</div>
        <div id="projectTree"></div>
      </div>
    </aside>
    <main>
      <header>
        <div class="title" id="conversationTitle">未选择对话</div>
        <div class="subtitle" id="projectSubtitle">请选择或添加一个项目</div>
      </header>
      <div class="conversation-body">
        <nav class="day-nav" id="dayNav" aria-label="聊天日期目录"></nav>
        <div class="messages" id="messages"></div>
      </div>
      <form class="composer" id="composer">
        <textarea id="messageInput" placeholder="输入你的需求"></textarea>
        <button id="sendButton" type="submit">发送</button>
      </form>
    </main>
  </div>
  <script>
    const state = {
      projects: [],
      conversations: [],
      running: {},
      projectId: "",
      conversationId: "",
    };
    const $ = (id) => document.getElementById(id);
    const pendingStarted = {};
    const expandedProjects = new Set();

    async function api(path, options = {}) {
      const response = await fetch(path, {
        headers: { "Content-Type": "application/json" },
        ...options,
      });
      const data = await response.json();
      if (!response.ok || data.error) throw new Error(data.error || response.statusText);
      return data;
    }

    async function loadState() {
      const data = await api("/api/state");
      state.projects = data.projects || [];
      state.conversations = data.conversations || [];
      state.running = data.running_conversations || {};
      state.projects.forEach((project) => expandedProjects.add(project.id));
      reconcileSelection();
      render();
    }

    function reconcileSelection() {
      if (!state.projects.some((project) => project.id === state.projectId)) {
        state.projectId = state.projects[0] ? state.projects[0].id : "";
      }
      const projectConversations = conversationsForProject(state.projectId);
      if (!projectConversations.some((conversation) => conversation.id === state.conversationId)) {
        state.conversationId = projectConversations[0] ? projectConversations[0].id : "";
      }
    }

    function conversationsForProject(projectId = state.projectId) {
      return state.conversations.filter((item) => item.project_id === projectId);
    }

    function currentProject() {
      return state.projects.find((item) => item.id === state.projectId);
    }

    function currentConversation() {
      return state.conversations.find((item) => item.id === state.conversationId);
    }

    function render() {
      const project = currentProject();
      const conversation = currentConversation();
      const running = conversation ? runningFor(conversation.id) : null;
      $("projectTree").innerHTML = renderProjectTree();
      $("currentProjectLabel").textContent = project ? projectLabel(project) : "未选择项目";
      $("conversationTitle").textContent = conversation ? conversation.title : "未选择对话";
      $("projectSubtitle").textContent = project ? project.workspace : "请选择或添加一个项目";
      $("sendButton").disabled = !conversation || Boolean(running);
      $("newConversation").disabled = !project;

      let messages = conversation ? [...conversation.messages] : [];
      if (conversation && running && lastRole(messages) === "user") {
        messages.push({
          role: "assistant",
          content: "正在思考",
          pending: true,
          started_at: running.started_at,
          events: running.events || [],
        });
      }
      const days = daysForMessages(messages);
      $("dayNav").innerHTML = renderSideNav(days, conversation);
      $("messages").innerHTML = renderConversationMessages(messages);
      $("messages").scrollTop = $("messages").scrollHeight;
      resizeComposer();
    }

    function renderProjectTree() {
      if (!state.projects.length) {
        return `<div class="empty-project">还没有项目</div>`;
      }
      return state.projects.map((project) => {
        const expanded = expandedProjects.has(project.id);
        const active = project.id === state.projectId ? " active" : "";
        const conversations = conversationsForProject(project.id);
        const conversationList = expanded
          ? `<div class="conversation-list">${
              conversations.length
                ? conversations.map(renderConversationRow).join("")
                : `<div class="empty-project">暂无对话</div>`
            }</div>`
          : "";
        return `
          <div class="project-group">
            <div class="project-row${active}">
              <button class="project-toggle" data-toggle-project="${project.id}" title="${expanded ? "收起项目" : "展开项目"}">${expanded ? "v" : ">"}</button>
              <button class="project-name" data-project-id="${project.id}" title="${escapeHtml(project.workspace)}">${escapeHtml(projectLabel(project))}</button>
              <button class="project-action" data-project-rename-id="${project.id}" title="重命名项目">改</button>
              <button class="project-action" data-project-new-id="${project.id}" title="新建对话">+</button>
            </div>
            ${conversationList}
          </div>`;
      }).join("");
    }

    function renderConversationRow(conversation) {
      const active = conversation.id === state.conversationId ? " active" : "";
      const pinned = conversation.pinned ? " pinned" : "";
      return `
        <div class="item-row">
          <button class="item${active}${pinned}" data-conversation-id="${conversation.id}" title="${escapeHtml(conversation.title)}">${escapeHtml(conversation.title)}</button>
          <button class="conversation-action" data-pin-id="${conversation.id}" title="${conversation.pinned ? "取消置顶" : "置顶"}">${conversation.pinned ? "下" : "顶"}</button>
          <button class="conversation-action" data-rename-id="${conversation.id}" title="重命名对话">改</button>
          <button class="conversation-action danger" data-delete-id="${conversation.id}" title="删除对话">删</button>
        </div>`;
    }

    function projectLabel(project) {
      return project.display_name || project.workspace.split(/[\\/]/).pop() || project.workspace;
    }

    $("browseProject").onclick = async () => {
      const project = await api("/api/projects/pick", {
        method: "POST",
        body: JSON.stringify({}),
      });
      state.projectId = project.id;
      state.conversationId = "";
      await loadState();
    };

    $("addProject").onclick = async () => {
      const path = $("projectPath").value.trim();
      if (!path) return;
      const project = await api("/api/projects", {
        method: "POST",
        body: JSON.stringify({ path }),
      });
      state.projectId = project.id;
      state.conversationId = "";
      await loadState();
    };

    $("newConversation").onclick = async () => {
      await createConversationForProject(state.projectId);
    };

    $("projectTree").onclick = async (event) => {
      const toggle = event.target.closest("button[data-toggle-project]");
      if (toggle) {
        const projectId = toggle.dataset.toggleProject;
        if (expandedProjects.has(projectId)) expandedProjects.delete(projectId);
        else expandedProjects.add(projectId);
        selectProject(projectId);
        return;
      }

      const projectButton = event.target.closest("button[data-project-id]");
      if (projectButton) {
        selectProject(projectButton.dataset.projectId);
        return;
      }

      const projectRename = event.target.closest("button[data-project-rename-id]");
      if (projectRename) {
        await renameProject(projectRename.dataset.projectRenameId);
        return;
      }

      const projectNew = event.target.closest("button[data-project-new-id]");
      if (projectNew) {
        await createConversationForProject(projectNew.dataset.projectNewId);
        return;
      }

      const pinButton = event.target.closest("button[data-pin-id]");
      if (pinButton) {
        await togglePinConversation(pinButton.dataset.pinId);
        return;
      }

      const renameButton = event.target.closest("button[data-rename-id]");
      if (renameButton) {
        await renameConversation(renameButton.dataset.renameId);
        return;
      }

      const deleteButton = event.target.closest("button[data-delete-id]");
      if (deleteButton) {
        await deleteConversation(deleteButton.dataset.deleteId);
        return;
      }

      const conversationButton = event.target.closest("button[data-conversation-id]");
      if (!conversationButton) return;
      const conversation = state.conversations.find((item) => item.id === conversationButton.dataset.conversationId);
      if (!conversation) return;
      state.projectId = conversation.project_id;
      state.conversationId = conversation.id;
      expandedProjects.add(conversation.project_id);
      render();
    };

    function selectProject(projectId) {
      state.projectId = projectId;
      expandedProjects.add(projectId);
      const conversations = conversationsForProject(projectId);
      state.conversationId = conversations[0] ? conversations[0].id : "";
      render();
    }

    async function createConversationForProject(projectId) {
      if (!projectId) return;
      const conversation = await api("/api/conversations", {
        method: "POST",
        body: JSON.stringify({ project_id: projectId, title: "新对话" }),
      });
      state.projectId = projectId;
      state.conversationId = conversation.id;
      expandedProjects.add(projectId);
      await loadState();
    }

    async function renameProject(projectId) {
      const project = state.projects.find((item) => item.id === projectId);
      if (!project) return;
      const current = projectLabel(project);
      const name = prompt("项目名称", current);
      if (!name || !name.trim()) return;
      const updated = await api("/api/projects/rename", {
        method: "POST",
        body: JSON.stringify({ project_id: project.id, display_name: name.trim() }),
      });
      const index = state.projects.findIndex((item) => item.id === updated.id);
      if (index >= 0) state.projects[index] = updated;
      state.projectId = updated.id;
      render();
    }

    async function renameConversation(conversationId) {
      const conversation = state.conversations.find((item) => item.id === conversationId);
      if (!conversation) return;
      const title = prompt("对话名称", conversation.title);
      if (!title || !title.trim()) return;
      const updated = await api("/api/conversations/rename", {
        method: "POST",
        body: JSON.stringify({ conversation_id: conversation.id, title: title.trim() }),
      });
      const index = state.conversations.findIndex((item) => item.id === updated.id);
      if (index >= 0) state.conversations[index] = updated;
      render();
    }

    async function togglePinConversation(conversationId) {
      const conversation = state.conversations.find((item) => item.id === conversationId);
      if (!conversation) return;
      const updated = await api("/api/conversations/pin", {
        method: "POST",
        body: JSON.stringify({ conversation_id: conversation.id, pinned: !conversation.pinned }),
      });
      updateConversation(updated);
      state.projectId = updated.project_id;
      state.conversationId = updated.id;
      render();
    }

    async function deleteConversation(conversationId) {
      const conversation = state.conversations.find((item) => item.id === conversationId);
      if (!conversation) return;
      if (!confirm(`删除对话“${conversation.title}”？`)) return;
      await api("/api/conversations/delete", {
        method: "POST",
        body: JSON.stringify({ conversation_id: conversation.id }),
      });
      state.conversations = state.conversations.filter((item) => item.id !== conversation.id);
      if (state.conversationId === conversation.id) {
        const remaining = conversationsForProject(conversation.project_id);
        state.conversationId = remaining[0] ? remaining[0].id : "";
      }
      render();
    }

    function updateConversation(updated) {
      const index = state.conversations.findIndex((item) => item.id === updated.id);
      if (index >= 0) state.conversations[index] = updated;
      else state.conversations.unshift(updated);
      state.conversations.sort((a, b) => {
        if (Boolean(a.pinned) !== Boolean(b.pinned)) return a.pinned ? -1 : 1;
        return String(b.updated_at || "").localeCompare(String(a.updated_at || ""));
      });
    }

    $("dayNav").onclick = (event) => {
      const nodeButton = event.target.closest("button[data-message-checkout]");
      if (nodeButton) {
        checkoutMessage(nodeButton.dataset.messageCheckout);
        return;
      }
      const button = event.target.closest("button[data-day]");
      if (!button) return;
      const target = document.getElementById(dayElementId(button.dataset.day));
      if (target) target.scrollIntoView({ block: "start", behavior: "smooth" });
    };

    $("composer").onsubmit = async (event) => {
      event.preventDefault();
      const content = $("messageInput").value.trim();
      if (!content || !state.conversationId) return;
      $("sendButton").disabled = true;
      $("messageInput").value = "";
      resizeComposer();
      appendOptimisticMessages(state.conversationId, content);
      render();
      try {
        const result = await api("/api/messages", {
          method: "POST",
          body: JSON.stringify({
            conversation_id: state.conversationId,
            content,
            parent_message_id: currentConversation()?.current_message_id || "",
          }),
        });
        const updated = result.conversation;
        updateConversation(updated);
        state.running[updated.id] = result.running;
      } catch (error) {
        markOptimisticFailure(state.conversationId, error.message);
      }
      render();
    };

    $("messageInput").addEventListener("focus", resizeComposer);
    $("messageInput").addEventListener("input", resizeComposer);
    $("messageInput").addEventListener("blur", resizeComposer);

    function resizeComposer() {
      const textarea = $("messageInput");
      const hasContent = textarea.value.trim().length > 0;
      const expanded = document.activeElement === textarea || hasContent;
      textarea.classList.toggle("expanded", expanded);
      textarea.style.height = expanded ? "84px" : "42px";
      if (!expanded) return;
      textarea.style.height = "auto";
      const next = Math.min(Math.max(textarea.scrollHeight, 84), 160);
      textarea.style.height = `${next}px`;
      textarea.style.overflowY = textarea.scrollHeight > 160 ? "auto" : "hidden";
    }

    function appendOptimisticMessages(conversationId, content) {
      const conversation = state.conversations.find((item) => item.id === conversationId);
      if (!conversation) return;
      const now = new Date().toISOString();
      const parentId = conversation.current_message_id || "";
      const userId = `pending-user-${Date.now()}`;
      const assistantId = `pending-assistant-${Date.now()}`;
      const previousMessages = conversation.messages || [];
      const previousTree = conversation.message_tree || previousMessages;
      pendingStarted[conversationId] = now;
      const userMessage = { id: userId, parent_id: parentId, role: "user", content, created_at: now, events: [] };
      const assistantMessage = {
        id: assistantId,
        parent_id: userId,
        role: "assistant",
        content: "正在思考",
        created_at: now,
        events: [],
        pending: true,
        started_at: now,
      };
      conversation.messages = [
        ...previousMessages,
        userMessage,
        assistantMessage,
      ];
      conversation.message_tree = [
        ...previousTree,
        userMessage,
        assistantMessage,
      ];
      conversation.current_message_id = userId;
      if (conversation.title === "新对话") conversation.title = content.slice(0, 32) || "新对话";
      state.running[conversationId] = { started_at: now, user_message_id: userId, events: [] };
    }

    function markOptimisticFailure(conversationId, reason) {
      const conversation = state.conversations.find((item) => item.id === conversationId);
      if (!conversation) return;
      conversation.messages = (conversation.messages || []).filter((message) => !message.pending);
      conversation.messages.push({
        role: "assistant",
        content: `任务失败: ${reason}`,
        created_at: new Date().toISOString(),
        events: [],
      });
      delete state.running[conversationId];
    }

    async function checkoutMessage(messageId) {
      const conversation = currentConversation();
      if (!conversation || !messageId || runningFor(conversation.id)) return;
      const updated = await api("/api/conversations/checkout", {
        method: "POST",
        body: JSON.stringify({ conversation_id: conversation.id, message_id: messageId }),
      });
      updateConversation(updated);
      state.projectId = updated.project_id;
      state.conversationId = updated.id;
      render();
    }

    function lastRole(messages) {
      if (!messages.length) return "";
      return messages[messages.length - 1].role;
    }

    function runningFor(conversationId) {
      const value = state.running[conversationId];
      if (!value) return null;
      if (typeof value === "string") return { started_at: value, events: [] };
      return value;
    }

    function elapsedSeconds(startedAt) {
      const started = Date.parse(startedAt || new Date().toISOString());
      return Math.max(0, Math.floor((Date.now() - started) / 1000));
    }

    function renderMessageContent(message) {
      if (message.role === "assistant") return renderMarkdown(message.content || "");
      return escapeHtml(message.content || "").replace(/\n/g, "<br>");
    }

    function renderConversationMessages(messages) {
      let currentDay = "";
      const parts = [];
      for (const message of messages) {
        const day = dayKey(message);
        if (day !== currentDay) {
          currentDay = day;
          parts.push(`<div class="day-separator" id="${dayElementId(day)}">${escapeHtml(dayLabel(day))}</div>`);
        }
        parts.push(renderMessage(message));
      }
      return parts.join("");
    }

    function renderMessage(message) {
      const events = renderEvents(message);
      const pending = message.pending ? " pending" : "";
      const elapsed = message.pending
        ? `<div class="elapsed">已处理 ${elapsedSeconds(message.started_at)} 秒</div>`
        : "";
      const content = renderMessageContent(message);
      return `<div class="message ${message.role}${pending}">${elapsed}<div class="message-content">${content}</div>${events}</div>`;
    }

    function renderEvents(message) {
      if (!message.events || !message.events.length) return "";
      const body = `<div class="events">${escapeHtml(message.events.join("\n"))}</div>`;
      if (message.pending) return body;
      return `<details class="events-details"><summary>执行过程 ${message.events.length} 条</summary>${body}</details>`;
    }

    function daysForMessages(messages) {
      const seen = new Set();
      const days = [];
      for (const message of messages) {
        const day = dayKey(message);
        if (seen.has(day)) continue;
        seen.add(day);
        days.push(day);
      }
      return days;
    }

    function renderSideNav(days, conversation) {
      return [
        `<div class="nav-group-title">日期</div>`,
        renderDayNav(days),
        `<div class="nav-group-title">Session 树</div>`,
        renderSessionTree(conversation),
      ].join("");
    }

    function renderDayNav(days) {
      if (!days.length) return `<div class="day-empty">暂无消息</div>`;
      return days.map((day) => (
        `<button class="day-link" data-day="${escapeHtml(day)}" title="${escapeHtml(dayLabel(day))}">${escapeHtml(shortDayLabel(day))}</button>`
      )).join("");
    }

    function renderSessionTree(conversation) {
      if (!conversation) return `<div class="day-empty">暂无节点</div>`;
      const nodes = conversation.message_tree || conversation.messages || [];
      if (!nodes.length) return `<div class="day-empty">暂无节点</div>`;
      const activeId = conversation.current_message_id || (conversation.messages.at(-1)?.id || "");
      const pathIds = new Set((conversation.messages || []).map((message) => message.id).filter(Boolean));
      const childCount = {};
      for (const node of nodes) {
        const parentId = node.parent_id || "";
        childCount[parentId] = (childCount[parentId] || 0) + 1;
      }
      const depths = messageDepths(nodes);
      return `<div class="session-tree">${
        nodes.map((message) => {
          const id = message.id || "";
          const active = id && id === activeId ? " active" : "";
          const onPath = pathIds.has(id) ? " on-path" : "";
          const depth = Math.min(depths[id] || 0, 6);
          const children = childCount[id] || 0;
          const branch = children > 1 ? ` <span class="branch-count">分叉 ${children}</span>` : "";
          const label = `${message.role === "user" ? "你" : "Agent"}：${messagePreview(message.content || "")}`;
          return `<button class="tree-node${active}${onPath}" data-message-checkout="${escapeHtml(id)}" style="padding-left:${8 + depth * 12}px" title="${escapeHtml(label)}"><span class="tree-label">${escapeHtml(label)}${branch}</span></button>`;
        }).join("")
      }</div>`;
    }

    function messageDepths(nodes) {
      const byId = {};
      for (const node of nodes) {
        if (node.id) byId[node.id] = node;
      }
      const cache = {};
      function depthOf(id, seen = new Set()) {
        if (!id || !byId[id] || seen.has(id)) return 0;
        if (cache[id] !== undefined) return cache[id];
        seen.add(id);
        const parentId = byId[id].parent_id || "";
        cache[id] = parentId && byId[parentId] ? depthOf(parentId, seen) + 1 : 0;
        return cache[id];
      }
      for (const node of nodes) {
        if (node.id) depthOf(node.id);
      }
      return cache;
    }

    function messagePreview(content) {
      return String(content).replace(/\s+/g, " ").trim().slice(0, 42) || "空消息";
    }

    function dayKey(message) {
      const raw = message.created_at || message.started_at || "";
      const timestamp = Date.parse(raw);
      if (Number.isNaN(timestamp)) return "unknown";
      return new Date(timestamp).toLocaleDateString("sv-SE");
    }

    function dayLabel(day) {
      if (day === "unknown") return "未知日期";
      return day;
    }

    function shortDayLabel(day) {
      if (day === "unknown") return "未知";
      return day.slice(5);
    }

    function dayElementId(day) {
      return `day-${String(day).replace(/[^a-zA-Z0-9_-]/g, "-")}`;
    }

    function renderMarkdown(markdown) {
      const lines = String(markdown).replace(/\r\n/g, "\n").split("\n");
      const blocks = [];
      let paragraph = [];
      let listItems = [];
      let codeLines = [];
      let inCode = false;

      function flushParagraph() {
        if (!paragraph.length) return;
        blocks.push(`<p>${renderInline(paragraph.join(" "))}</p>`);
        paragraph = [];
      }

      function flushList() {
        if (!listItems.length) return;
        blocks.push(`<ul>${listItems.map((item) => `<li>${renderInline(item)}</li>`).join("")}</ul>`);
        listItems = [];
      }

      function flushCode() {
        blocks.push(`<pre><code>${escapeHtml(codeLines.join("\n"))}</code></pre>`);
        codeLines = [];
      }

      for (const rawLine of lines) {
        const line = rawLine.trimEnd();
        if (line.trim().startsWith("```")) {
          if (inCode) {
            flushCode();
            inCode = false;
          } else {
            flushParagraph();
            flushList();
            inCode = true;
            codeLines = [];
          }
          continue;
        }
        if (inCode) {
          codeLines.push(rawLine);
          continue;
        }
        if (!line.trim()) {
          flushParagraph();
          flushList();
          continue;
        }
        const heading = line.match(/^(#{2,3})\s+(.+)$/);
        if (heading) {
          flushParagraph();
          flushList();
          const level = heading[1].length;
          blocks.push(`<h${level}>${renderInline(heading[2])}</h${level}>`);
          continue;
        }
        const bullet = line.match(/^\s*[-*]\s+(.+)$/);
        if (bullet) {
          flushParagraph();
          listItems.push(bullet[1]);
          continue;
        }
        flushList();
        paragraph.push(line.trim());
      }
      if (inCode) flushCode();
      flushParagraph();
      flushList();
      return blocks.join("");
    }

    function renderInline(value) {
      return escapeHtml(value)
        .replace(/`([^`]+)`/g, "<code>$1</code>")
        .replace(/\*\*([^*]+)\*\*/g, "<strong>$1</strong>");
    }

    let polling = false;
    async function pollState() {
      if (polling || !Object.keys(state.running).length) {
        if (currentConversation() && runningFor(state.conversationId)) render();
        return;
      }
      polling = true;
      try {
        await loadState();
      } catch (error) {
        console.warn(error);
        render();
      } finally {
        polling = false;
      }
    }

    setInterval(() => {
      pollState();
    }, 1000);

    function escapeHtml(value) {
      return String(value)
        .replaceAll("&", "&amp;")
        .replaceAll("<", "&lt;")
        .replaceAll(">", "&gt;")
        .replaceAll('"', "&quot;");
    }

    loadState().catch((error) => alert(error.message));
  </script>
</body>
</html>
"""


if __name__ == "__main__":
    raise SystemExit(main())
