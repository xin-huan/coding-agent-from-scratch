"""Minimal local chat UI for the coding agent."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
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
from coding_agent.workspace_snapshot import (
    WorkspaceSnapshotError,
    WorkspaceSnapshotStore,
)


@dataclass
class ChatMessage:
    role: str
    content: str
    events: list[str] = field(default_factory=list)
    created_at: str = ""
    id: str = ""
    parent_id: str = ""
    workspace_snapshot_id: str = ""

    def to_data(self) -> dict[str, object]:
        return {
            "id": self.id,
            "parent_id": self.parent_id,
            "role": self.role,
            "content": self.content,
            "events": self.events,
            "created_at": self.created_at,
            "workspace_snapshot_id": self.workspace_snapshot_id,
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
            workspace_snapshot_id=str(data.get("workspace_snapshot_id", "")),
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

    def message_by_id(self, message_id: str) -> ChatMessage:
        for message in self.messages:
            if message.id == message_id:
                return message
        raise KeyError(f"Message not found: {message_id}")

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

    def delete_project_conversations(self, project_id: str) -> list[str]:
        deleted = [
            conversation_id
            for conversation_id, conversation in self.conversations.items()
            if conversation.project_id == project_id
        ]
        for conversation_id in deleted:
            del self.conversations[conversation_id]
        if deleted:
            self.save()
        return deleted

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
        workspace_snapshot_id: str = "",
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
            workspace_snapshot_id=workspace_snapshot_id,
        )
        conversation.messages.append(message)
        conversation.current_message_id = message.id
        conversation.updated_at = now
        self.save()
        return message

    def set_message_snapshot(
        self,
        conversation_id: str,
        message_id: str,
        snapshot_id: str,
    ) -> ChatMessage:
        conversation = self.get(conversation_id)
        message = conversation.message_by_id(message_id)
        message.workspace_snapshot_id = snapshot_id
        conversation.updated_at = _now()
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
        self.snapshot_store = WorkspaceSnapshotStore(data_dir / "workspace-snapshots")
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
                    "sort_order": memory.sort_order,
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
        memory_path = self.memory_store.path_for(workspace)
        is_new_project = not memory_path.exists()
        memory = self.memory_store.load(workspace)
        memory.workspace = str(workspace.root)
        if not memory.display_name:
            memory.display_name = workspace.root.name or str(workspace.root)
        if is_new_project:
            projects = self.memory_store.list_projects()
            memory.sort_order = max((project.sort_order for project in projects), default=-1) + 1
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

    def delete_project(self, project_id: str) -> dict[str, object]:
        with self._lock:
            conversations = [
                conversation
                for conversation in self.chat_store.list_conversations()
                if conversation.project_id == project_id
            ]
            running = [
                conversation.id
                for conversation in conversations
                if conversation.id in self._running_conversations
            ]
            if running:
                raise RuntimeError("Cannot delete a project while its conversation is running")
            deleted_conversations = self.chat_store.delete_project_conversations(project_id)
            for conversation_id in deleted_conversations:
                self._agents.pop(conversation_id, None)
            self.memory_store.delete_project(project_id)
        return {
            "deleted": project_id,
            "deleted_conversations": deleted_conversations,
        }

    def reorder_projects(self, project_ids: list[str]) -> dict[str, object]:
        with self._lock:
            projects = self.memory_store.reorder_projects(project_ids)
        return {
            "projects": [
                {
                    "id": memory.project_id,
                    "workspace": memory.workspace,
                    "display_name": memory.display_name,
                    "sort_order": memory.sort_order,
                    "updated_at": memory.updated_at,
                }
                for memory in projects
            ]
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

    def restore_workspace_snapshot(
        self,
        conversation_id: str,
        snapshot_id: str,
    ) -> dict[str, object]:
        with self._lock:
            if conversation_id in self._running_conversations:
                raise RuntimeError("Cannot restore code while a conversation is running")
            conversation = self.chat_store.get(conversation_id)
            matching_message = next(
                (
                    message
                    for message in conversation.messages
                    if message.workspace_snapshot_id == snapshot_id
                ),
                None,
            )
            if matching_message is None:
                raise WorkspaceSnapshotError("Snapshot does not belong to this conversation")
            workspace = Workspace(Path(self._workspace_for_project(conversation.project_id)))
            result = self.snapshot_store.restore(workspace, snapshot_id, created_at=_now())
            self._agents.pop(conversation_id, None)
        return result.to_data()

    def send_message(self, conversation_id: str, content: str) -> dict[str, object]:
        conversation = self.chat_store.get(conversation_id)
        workspace_path = self._workspace_for_project(conversation.project_id)
        workspace = Workspace(Path(workspace_path))
        before_state = self.snapshot_store.capture_state(workspace)
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
        assistant_message = self.chat_store.append_assistant_message(
            conversation.id,
            content=answer,
            events=events,
            parent_id=user_message.id,
        )
        self._capture_workspace_snapshot(
            workspace,
            before_state=before_state,
            conversation_id=conversation.id,
            project_id=conversation.project_id,
            assistant_message_id=assistant_message.id,
            user_message_id=user_message.id,
            created_at=assistant_message.created_at,
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
            before_state = self.snapshot_store.capture_state(workspace)
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
                assistant_message = self.chat_store.append_assistant_message(
                    conversation_id,
                    content=answer,
                    events=events,
                    parent_id=user_message_id,
                )
                conversation = self.chat_store.get(conversation_id)
                if "workspace" in locals() and "before_state" in locals():
                    self._capture_workspace_snapshot(
                        workspace,
                        before_state=before_state,
                        conversation_id=conversation_id,
                        project_id=conversation.project_id,
                        assistant_message_id=assistant_message.id,
                        user_message_id=user_message_id,
                        created_at=assistant_message.created_at,
                    )
                self._running_conversations.pop(conversation_id, None)

    def _capture_workspace_snapshot(
        self,
        workspace: Workspace,
        *,
        before_state: dict[str, str],
        conversation_id: str,
        project_id: str,
        assistant_message_id: str,
        user_message_id: str,
        created_at: str,
    ) -> str:
        after_state = self.snapshot_store.capture_state(workspace)
        snapshot = self.snapshot_store.create_for_changes(
            before=before_state,
            after=after_state,
            project_id=project_id,
            conversation_id=conversation_id,
            message_id=assistant_message_id,
            user_message_id=user_message_id,
            created_at=created_at,
        )
        if snapshot is None:
            return ""
        self.chat_store.set_message_snapshot(
            conversation_id,
            assistant_message_id,
            snapshot.id,
        )
        return snapshot.id

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
                elif parsed.path == "/api/projects/delete":
                    result = app.delete_project(
                        str(payload.get("project_id", "")),
                    )
                elif parsed.path == "/api/projects/reorder":
                    project_ids = payload.get("project_ids", [])
                    if not isinstance(project_ids, list):
                        raise ValueError("project_ids must be a list")
                    result = app.reorder_projects([str(item) for item in project_ids])
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
                elif parsed.path == "/api/conversations/restore-snapshot":
                    result = app.restore_workspace_snapshot(
                        str(payload.get("conversation_id", "")),
                        str(payload.get("snapshot_id", "")),
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
            except (
                KeyError,
                RuntimeError,
                ValueError,
                WorkspaceError,
                WorkspaceSnapshotError,
            ) as error:
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
    errors: list[str] = []
    child_code = (
        "import sys\n"
        "try:\n"
        "    import tkinter as tk\n"
        "    from tkinter import filedialog\n"
        "    root = tk.Tk()\n"
        "    root.withdraw()\n"
        "    root.attributes('-topmost', True)\n"
        "    root.lift()\n"
        "    root.update()\n"
        "    selected = filedialog.askdirectory(parent=root, title='选择项目文件夹', mustexist=True)\n"
        "    root.destroy()\n"
        "    sys.stdout.reconfigure(encoding='utf-8', errors='replace')\n"
        "    print(selected or '', flush=True)\n"
        "except Exception as error:\n"
        "    print(f'ERROR: {error}', file=sys.stderr, flush=True)\n"
        "    raise\n"
    )
    try:
        result = subprocess.run(
            [sys.executable, "-c", child_code],
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=60,
            check=False,
        )
        if result.returncode == 0:
            return result.stdout.strip()
        errors.append(f"python-tk: {result.stderr.strip() or result.returncode}")
    except Exception as error:
        errors.append(f"python-tk: {error}")

    powershell = Path(r"C:\Windows\System32\WindowsPowerShell\v1.0\powershell.exe")
    if powershell.exists():
        shell_script = (
            "[Console]::OutputEncoding = [System.Text.Encoding]::UTF8;"
            "$shell = New-Object -ComObject Shell.Application;"
            "$folder = $shell.BrowseForFolder(0, '选择项目文件夹', 0x41, 17);"
            "if ($folder -ne $null) { Write-Output $folder.Self.Path }"
        )
        try:
            result = subprocess.run(
                [str(powershell), "-NoProfile", "-STA", "-Command", shell_script],
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
                timeout=60,
                check=False,
            )
            if result.returncode == 0:
                selected = result.stdout.strip()
                if selected:
                    return selected
                return ""
            errors.append(f"shell: {result.stderr.strip() or result.returncode}")
        except Exception as error:
            errors.append(f"shell: {error}")

        script = (
            "Add-Type -AssemblyName System.Windows.Forms;"
            "$owner = New-Object System.Windows.Forms.Form;"
            "$owner.TopMost = $true;"
            "$owner.ShowInTaskbar = $false;"
            "$owner.StartPosition = 'CenterScreen';"
            "$owner.Width = 1;"
            "$owner.Height = 1;"
            "$dialog = New-Object System.Windows.Forms.FolderBrowserDialog;"
            "$dialog.Description = '选择项目文件夹';"
            "$dialog.ShowNewFolderButton = $true;"
            "$owner.Show();"
            "$result = $dialog.ShowDialog($owner);"
            "$owner.Dispose();"
            "if ($result -eq [System.Windows.Forms.DialogResult]::OK) {"
            "  [Console]::OutputEncoding = [System.Text.Encoding]::UTF8;"
            "  Write-Output $dialog.SelectedPath"
            "}"
        )
        try:
            result = subprocess.run(
                [str(powershell), "-NoProfile", "-STA", "-Command", script],
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
                timeout=60,
                check=False,
            )
            if result.returncode == 0:
                return result.stdout.strip()
            errors.append(f"powershell: {result.stderr.strip() or result.returncode}")
        except Exception as error:
            errors.append(f"powershell: {error}")

    raise RuntimeError("Directory picker is not available: " + "; ".join(errors))


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
      grid-template-columns: var(--sidebar-width, 320px) 6px minmax(0, 1fr);
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
      grid-template-columns: auto minmax(0, 1fr) auto auto auto;
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
    .project-group.dragging {
      opacity: .5;
    }
    .project-group.drop-target .project-row {
      border-top-color: var(--accent);
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
    .project-action.danger:hover,
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
      grid-template-columns: var(--nav-width, 180px) 6px minmax(0, 1fr);
      min-height: 0;
    }
    .resize-handle {
      background: transparent;
      position: relative;
      z-index: 5;
    }
    .resize-handle.vertical {
      cursor: col-resize;
    }
    .resize-handle.vertical::after {
      content: "";
      position: absolute;
      top: 0;
      bottom: 0;
      left: 2px;
      width: 1px;
      background: var(--line);
    }
    .resize-handle:hover::after,
    body.resizing .resize-handle::after {
      background: var(--accent);
      width: 2px;
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
    .tree-entry {
      margin-left: var(--tree-indent, 0px);
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
      position: relative;
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
    .tree-node::before {
      content: "";
      position: absolute;
      left: -2px;
      top: -3px;
      bottom: -3px;
      border-left: 2px solid var(--line);
    }
    .tree-node.on-path::before {
      border-left-color: #c7d7d4;
    }
    .tree-node.active::before {
      border-left-color: var(--accent);
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
    .snapshot-badge {
      color: var(--accent-strong);
      font-size: 11px;
      margin-left: 4px;
    }
    .turn-menu {
      position: fixed;
      z-index: 20;
      display: grid;
      gap: 6px;
      min-width: 132px;
      padding: 8px;
      border: 1px solid var(--line);
      border-radius: 8px;
      background: #fff;
      box-shadow: 0 12px 30px rgba(15, 23, 42, .16);
    }
    .turn-menu[hidden] {
      display: none;
    }
    .turn-menu-title {
      max-width: 220px;
      color: var(--muted);
      font-size: 11px;
      overflow: hidden;
      text-overflow: ellipsis;
      white-space: nowrap;
      padding: 0 2px 2px;
    }
    .turn-menu-actions {
      display: flex;
      flex-direction: column;
      gap: 6px;
    }
    .turn-action {
      border: 1px solid var(--line);
      border-radius: 8px;
      background: #fff;
      color: var(--text);
      padding: 6px 9px;
      font-size: 12px;
    }
    .turn-action.primary {
      border-color: var(--accent);
      color: var(--accent-strong);
    }
    .turn-action:disabled {
      color: var(--muted);
      background: #f3f6f5;
      cursor: not-allowed;
    }
    .tree-node:hover {
      color: var(--accent-strong);
      border-left-color: var(--accent);
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
      scroll-margin-top: 16px;
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
      .app-resizer,
      .nav-resizer { display: none; }
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
    <div class="resize-handle vertical app-resizer" id="appResizer" title="拖动调整项目栏宽度"></div>
    <main>
      <header>
        <div class="title" id="conversationTitle">未选择对话</div>
        <div class="subtitle" id="projectSubtitle">请选择或添加一个项目</div>
      </header>
      <div class="conversation-body">
        <nav class="day-nav" id="dayNav" aria-label="聊天日期目录"></nav>
        <div class="resize-handle vertical nav-resizer" id="navResizer" title="拖动调整目录宽度"></div>
        <div class="messages" id="messages"></div>
      </div>
      <form class="composer" id="composer">
        <textarea id="messageInput" placeholder="输入你的需求"></textarea>
        <button id="sendButton" type="submit">发送</button>
      </form>
    </main>
  </div>
  <div class="turn-menu" id="turnMenu" hidden></div>
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
    let turnMenuState = null;
    let draggedProjectId = "";
    let projectsInitialized = false;
    const layoutKeys = {
      sidebar: "coding-agent.sidebar-width",
      nav: "coding-agent.nav-width",
    };

    function clamp(value, min, max) {
      return Math.min(Math.max(value, min), max);
    }

    function loadLayout() {
      const sidebar = Number(localStorage.getItem(layoutKeys.sidebar));
      const nav = Number(localStorage.getItem(layoutKeys.nav));
      if (Number.isFinite(sidebar) && sidebar > 0) {
        document.documentElement.style.setProperty("--sidebar-width", `${clamp(sidebar, 240, 520)}px`);
      }
      if (Number.isFinite(nav) && nav > 0) {
        document.documentElement.style.setProperty("--nav-width", `${clamp(nav, 120, 380)}px`);
      }
    }

    function setupResizer(handleId, options) {
      const handle = $(handleId);
      if (!handle) return;
      handle.addEventListener("pointerdown", (event) => {
        if (window.matchMedia("(max-width: 820px)").matches) return;
        event.preventDefault();
        handle.setPointerCapture(event.pointerId);
        document.body.classList.add("resizing");
        const startX = event.clientX;
        const startWidth = options.currentWidth();

        function move(moveEvent) {
          const nextWidth = clamp(startWidth + moveEvent.clientX - startX, options.min(), options.max());
          document.documentElement.style.setProperty(options.cssVariable, `${nextWidth}px`);
          localStorage.setItem(options.storageKey, String(Math.round(nextWidth)));
        }

        function finish(upEvent) {
          document.body.classList.remove("resizing");
          handle.releasePointerCapture(upEvent.pointerId);
          handle.removeEventListener("pointermove", move);
          handle.removeEventListener("pointerup", finish);
          handle.removeEventListener("pointercancel", finish);
        }

        handle.addEventListener("pointermove", move);
        handle.addEventListener("pointerup", finish);
        handle.addEventListener("pointercancel", finish);
      });
    }

    function setupResizers() {
      setupResizer("appResizer", {
        cssVariable: "--sidebar-width",
        storageKey: layoutKeys.sidebar,
        currentWidth: () => document.querySelector("aside").getBoundingClientRect().width,
        min: () => 240,
        max: () => Math.max(260, window.innerWidth - 520),
      });
      setupResizer("navResizer", {
        cssVariable: "--nav-width",
        storageKey: layoutKeys.nav,
        currentWidth: () => $("dayNav").getBoundingClientRect().width,
        min: () => 120,
        max: () => Math.min(420, Math.max(160, window.innerWidth - 520)),
      });
    }

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
      if (!projectsInitialized) {
        state.projects.forEach((project) => expandedProjects.add(project.id));
        projectsInitialized = true;
      }
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
          <div class="project-group" draggable="true" data-project-drag-id="${project.id}">
            <div class="project-row${active}">
              <button class="project-toggle" data-toggle-project="${project.id}" title="${expanded ? "收起项目" : "展开项目"}">${expanded ? "v" : ">"}</button>
              <button class="project-name" data-project-id="${project.id}" title="${escapeHtml(project.workspace)}">${escapeHtml(projectLabel(project))}</button>
              <button class="project-action" data-project-rename-id="${project.id}" title="重命名项目">改</button>
              <button class="project-action" data-project-new-id="${project.id}" title="新建对话">+</button>
              <button class="project-action danger" data-project-delete-id="${project.id}" title="移除项目">删</button>
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
      const button = $("browseProject");
      const previousText = button.textContent;
      button.disabled = true;
      button.textContent = "选择中...";
      try {
        const project = await api("/api/projects/pick", {
          method: "POST",
          body: JSON.stringify({}),
        });
        state.projectId = project.id;
        state.conversationId = "";
        expandedProjects.add(project.id);
        hideTurnMenu();
        await loadState();
      } catch (error) {
        alert(`选择项目失败: ${error.message}`);
      } finally {
        button.disabled = false;
        button.textContent = previousText;
      }
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
      expandedProjects.add(project.id);
      hideTurnMenu();
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
        focusProject(projectId);
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

      const projectDelete = event.target.closest("button[data-project-delete-id]");
      if (projectDelete) {
        await deleteProject(projectDelete.dataset.projectDeleteId);
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
      hideTurnMenu();
      expandedProjects.add(conversation.project_id);
      render();
    };

    $("projectTree").addEventListener("dragstart", (event) => {
      const group = event.target.closest("[data-project-drag-id]");
      if (!group) return;
      draggedProjectId = group.dataset.projectDragId || "";
      group.classList.add("dragging");
      event.dataTransfer.effectAllowed = "move";
      event.dataTransfer.setData("text/plain", draggedProjectId);
    });

    $("projectTree").addEventListener("dragover", (event) => {
      const group = event.target.closest("[data-project-drag-id]");
      if (!group || !draggedProjectId || group.dataset.projectDragId === draggedProjectId) return;
      event.preventDefault();
      event.dataTransfer.dropEffect = "move";
      document.querySelectorAll(".project-group.drop-target").forEach((item) => item.classList.remove("drop-target"));
      group.classList.add("drop-target");
    });

    $("projectTree").addEventListener("drop", async (event) => {
      const group = event.target.closest("[data-project-drag-id]");
      if (!group || !draggedProjectId) return;
      event.preventDefault();
      const targetProjectId = group.dataset.projectDragId || "";
      const rect = group.getBoundingClientRect();
      const placeAfter = event.clientY > rect.top + rect.height / 2;
      document.querySelectorAll(".project-group.drop-target").forEach((item) => item.classList.remove("drop-target"));
      await moveProjectTo(draggedProjectId, targetProjectId, placeAfter);
      draggedProjectId = "";
    });

    $("projectTree").addEventListener("dragend", () => {
      draggedProjectId = "";
      document.querySelectorAll(".project-group.dragging, .project-group.drop-target").forEach((item) => {
        item.classList.remove("dragging", "drop-target");
      });
    });

    async function moveProjectTo(projectId, targetProjectId, placeAfter = false) {
      if (!projectId || !targetProjectId || projectId === targetProjectId) return;
      const ids = state.projects.map((project) => project.id).filter((id) => id !== projectId);
      const targetIndex = ids.indexOf(targetProjectId);
      if (targetIndex < 0) return;
      ids.splice(placeAfter ? targetIndex + 1 : targetIndex, 0, projectId);
      state.projects = ids
        .map((id) => state.projects.find((project) => project.id === id))
        .filter(Boolean);
      render();
      const result = await api("/api/projects/reorder", {
        method: "POST",
        body: JSON.stringify({ project_ids: ids }),
      });
      state.projects = result.projects || state.projects;
      render();
    }

    function selectProject(projectId) {
      state.projectId = projectId;
      hideTurnMenu();
      expandedProjects.add(projectId);
      const conversations = conversationsForProject(projectId);
      state.conversationId = conversations[0] ? conversations[0].id : "";
      render();
    }

    function focusProject(projectId) {
      state.projectId = projectId;
      const conversations = conversationsForProject(projectId);
      if (!conversations.some((conversation) => conversation.id === state.conversationId)) {
        state.conversationId = conversations[0] ? conversations[0].id : "";
      }
      hideTurnMenu();
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
      hideTurnMenu();
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

    async function deleteProject(projectId) {
      const project = state.projects.find((item) => item.id === projectId);
      if (!project) return;
      const name = projectLabel(project);
      if (!confirm(`从列表移除项目“${name}”？这只会删除 Agent 的项目记录和对话历史，不会删除磁盘上的项目文件。`)) return;
      await api("/api/projects/delete", {
        method: "POST",
        body: JSON.stringify({ project_id: project.id }),
      });
      state.projects = state.projects.filter((item) => item.id !== project.id);
      state.conversations = state.conversations.filter((item) => item.project_id !== project.id);
      expandedProjects.delete(project.id);
      if (state.projectId === project.id) {
        state.projectId = state.projects[0] ? state.projects[0].id : "";
        if (state.projectId) expandedProjects.add(state.projectId);
        const conversations = conversationsForProject(state.projectId);
        state.conversationId = conversations[0] ? conversations[0].id : "";
        hideTurnMenu();
      }
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
      const continueButton = event.target.closest("button[data-turn-continue]");
      if (continueButton) {
        continueFromTurn(continueButton.dataset.turnContinue, continueButton.dataset.turnUser || "");
        return;
      }
      const restoreButton = event.target.closest("button[data-turn-restore]");
      if (restoreButton) {
        restoreSnapshotForTurn(
          restoreButton.dataset.turnRestore || "",
          restoreButton.dataset.turnLabel || restoreButton.textContent || "这个节点",
        );
        return;
      }
      const nodeButton = event.target.closest("button[data-turn-select]");
      if (nodeButton) {
        hideTurnMenu();
        scrollToMessage(nodeButton.dataset.turnUser || "");
        return;
      }
      const button = event.target.closest("button[data-day]");
      if (!button) return;
      const target = document.getElementById(dayElementId(button.dataset.day));
      if (target) target.scrollIntoView({ block: "start", behavior: "smooth" });
    };

    $("dayNav").addEventListener("contextmenu", (event) => {
      const nodeButton = event.target.closest("button[data-turn-select]");
      if (!nodeButton) return;
      event.preventDefault();
      openTurnMenu(nodeButton, event.clientX, event.clientY);
    });

    $("turnMenu").onclick = (event) => {
      const continueButton = event.target.closest("button[data-turn-continue]");
      if (continueButton) {
        continueFromTurn(continueButton.dataset.turnContinue, continueButton.dataset.turnUser || "");
        return;
      }
      const restoreButton = event.target.closest("button[data-turn-restore]");
      if (restoreButton) {
        restoreSnapshotForTurn(
          restoreButton.dataset.turnRestore || "",
          restoreButton.dataset.turnLabel || restoreButton.textContent || "这个节点",
        );
      }
    };

    document.addEventListener("click", (event) => {
      if ($("turnMenu").contains(event.target)) return;
      if (event.target.closest("button[data-turn-select]")) return;
      hideTurnMenu();
    });

    $("composer").onsubmit = async (event) => {
      event.preventDefault();
      const content = $("messageInput").value.trim();
      if (!content || !state.conversationId) return;
      const parentMessageId = currentConversation()?.current_message_id || "";
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
            parent_message_id: parentMessageId,
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

    function openTurnMenu(nodeButton, clientX, clientY) {
      const conversation = currentConversation();
      if (!conversation) return;
      const userId = nodeButton.dataset.turnUser || "";
      if (!userId) return;
      turnMenuState = {
        conversationId: conversation.id,
        userId,
        resumeId: nodeButton.dataset.messageCheckout || userId,
        snapshotId: nodeButton.dataset.snapshotId || "",
        label: nodeButton.dataset.turnLabel || nodeButton.textContent || "这个节点",
      };
      const menu = $("turnMenu");
      const restoreDisabled = turnMenuState.snapshotId ? "" : " disabled";
      menu.innerHTML = `
        <div class="turn-menu-title">${escapeHtml(turnMenuState.label)}</div>
        <div class="turn-menu-actions">
          <button class="turn-action primary" data-turn-continue="${escapeHtml(turnMenuState.resumeId)}" data-turn-user="${escapeHtml(userId)}">从此继续</button>
          <button class="turn-action" data-turn-restore="${escapeHtml(turnMenuState.snapshotId)}" data-turn-label="${escapeHtml(turnMenuState.label)}"${restoreDisabled}>恢复快照</button>
        </div>`;
      menu.hidden = false;
      const width = 180;
      const height = turnMenuState.snapshotId ? 104 : 104;
      menu.style.left = `${Math.min(clientX, window.innerWidth - width - 12)}px`;
      menu.style.top = `${Math.min(clientY, window.innerHeight - height - 12)}px`;
      scrollToMessage(userId);
    }

    function hideTurnMenu() {
      turnMenuState = null;
      const menu = $("turnMenu");
      if (menu) menu.hidden = true;
    }

    async function continueFromTurn(messageId, scrollMessageId = "") {
      hideTurnMenu();
      await checkoutMessage(messageId, scrollMessageId || messageId);
    }

    async function checkoutMessage(messageId, scrollMessageId = "") {
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
      scrollToMessage(scrollMessageId || messageId);
    }

    async function restoreSnapshotForTurn(snapshotId, label) {
      const conversation = currentConversation();
      if (!conversation || runningFor(conversation.id)) return;
      if (!snapshotId) {
        alert("这个节点没有可恢复的代码快照。只有 Agent 修改过文件的回合才会生成快照。");
        return;
      }
      const confirmed = confirm(
        `把代码恢复到“${label}”这轮 Agent 运行完毕时的状态？\n\n恢复前会先备份当前相关文件，但被恢复路径上的当前改动会被覆盖。`,
      );
      if (!confirmed) return;
      try {
        const result = await api("/api/conversations/restore-snapshot", {
          method: "POST",
          body: JSON.stringify({
            conversation_id: conversation.id,
            snapshot_id: snapshotId,
          }),
        });
        alert(`已恢复 ${result.restored_files.length} 个文件。当前状态已备份为 ${result.backup_id}。`);
        await loadState();
      } catch (error) {
        alert(`恢复失败: ${error.message}`);
      }
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
      const anchor = message.id ? ` id="${messageElementId(message.id)}"` : "";
      return `<div${anchor} class="message ${message.role}${pending}">${elapsed}<div class="message-content">${content}</div>${events}</div>`;
    }

    function scrollToMessage(messageId) {
      if (!messageId) return;
      requestAnimationFrame(() => {
        const target = document.getElementById(messageElementId(messageId));
        if (target) target.scrollIntoView({ block: "start", behavior: "smooth" });
      });
    }

    function messageElementId(messageId) {
      return `message-${String(messageId).replace(/[^a-zA-Z0-9_-]/g, "-")}`;
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
      const pathIds = new Set((conversation.messages || [])
        .filter((message) => message.role === "user")
        .map((message) => message.id)
        .filter(Boolean));
      const childrenByParent = {};
      for (const node of nodes) {
        const parentId = node.parent_id || "";
        if (!childrenByParent[parentId]) childrenByParent[parentId] = [];
        childrenByParent[parentId].push(node);
      }
      function replyForUser(user) {
        return (childrenByParent[user.id] || []).find((child) => child.role === "assistant") || null;
      }
      function resumeIdForUser(user) {
        const reply = replyForUser(user);
        return reply && reply.id ? reply.id : user.id;
      }
      const allTurns = nodes.filter((message) => message.role === "user");
      const userChildrenByParent = {};
      for (const message of allTurns) {
        const parentId = message.parent_id || "";
        if (!userChildrenByParent[parentId]) userChildrenByParent[parentId] = [];
        userChildrenByParent[parentId].push(message);
      }
      const turnDepths = {};
      function visibleDepthForTurn(message) {
        if (!message || !message.id) return 0;
        if (turnDepths[message.id] !== undefined) return turnDepths[message.id];
        const parent = nodes.find((node) => node.id === message.parent_id) || null;
        const parentTurn = parent && parent.role === "assistant"
          ? nodes.find((node) => node.id === parent.parent_id && node.role === "user")
          : null;
        const siblingCount = (userChildrenByParent[message.parent_id || ""] || []).length;
        const parentDepth = parentTurn ? visibleDepthForTurn(parentTurn) : 0;
        turnDepths[message.id] = parentTurn && siblingCount > 1 ? parentDepth + 1 : parentDepth;
        return turnDepths[message.id];
      }
      return `<div class="session-tree">${
        allTurns.map((message) => {
          const userId = message.id || "";
          const resumeId = resumeIdForUser(message) || userId;
          const active = resumeId && (resumeId === activeId || userId === activeId) ? " active" : "";
          const onPath = pathIds.has(userId) ? " on-path" : "";
          const branchChildren = (childrenByParent[resumeId || userId] || [])
            .filter((child) => child.role === "user" && child.id);
          const branch = branchChildren.length > 1 ? ` <span class="branch-count">分支 ${branchChildren.length}</span>` : "";
          const label = `你：${messagePreview(message.content || "")}`;
          const reply = replyForUser(message);
          const snapshotId = reply?.workspace_snapshot_id || "";
          const snapshotBadge = snapshotId ? ` <span class="snapshot-badge">快照</span>` : "";
          const depth = Math.min(visibleDepthForTurn(message), 6);
          return `<div class="tree-entry" style="--tree-indent:${depth * 14}px"><button class="tree-node${active}${onPath}" data-turn-select="${escapeHtml(userId)}" data-turn-user="${escapeHtml(userId)}" data-message-checkout="${escapeHtml(resumeId)}" data-message-scroll="${escapeHtml(userId)}" data-snapshot-id="${escapeHtml(snapshotId)}" data-turn-label="${escapeHtml(label)}" title="${escapeHtml(label)}"><span class="tree-label">${escapeHtml(label)}${branch}${snapshotBadge}</span></button></div>`;
        }).join("")
      }</div>`;
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

    loadLayout();
    setupResizers();
    loadState().catch((error) => alert(error.message));
  </script>
</body>
</html>
"""


if __name__ == "__main__":
    raise SystemExit(main())
