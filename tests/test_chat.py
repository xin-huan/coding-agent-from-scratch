import tempfile
import threading
import time
import unittest
from pathlib import Path

from coding_agent.chat import INDEX_HTML, ChatApp, ChatStore
from coding_agent.config import Settings


class FakeChatAgent:
    def __init__(self, on_event) -> None:
        self.on_event = on_event

    def run(self, task: str) -> str:
        self.on_event("[状态] 正在检查项目")
        self.on_event("[工具] list_files")
        return f"已处理：{task}"


class InspectingChatAgent:
    def __init__(self, on_event, inspect_state) -> None:
        self.on_event = on_event
        self.inspect_state = inspect_state

    def run(self, task: str) -> str:
        self.inspect_state(task)
        self.on_event("[状态] 正在检查项目")
        return "完成"


class SlowChatAgent:
    def __init__(
        self,
        on_event,
        started: threading.Event,
        finish: threading.Event,
    ) -> None:
        self.on_event = on_event
        self.started = started
        self.finish = finish

    def run(self, task: str) -> str:
        self.on_event("[状态] 正在检查项目")
        self.started.set()
        if not self.finish.wait(timeout=5):
            raise RuntimeError("slow test timed out")
        self.on_event("[工具] list_files")
        return f"后台完成：{task}"


class FakeChatApp(ChatApp):
    def _create_agent(self, workspace, on_event):
        return FakeChatAgent(on_event)


class InspectingChatApp(ChatApp):
    def __init__(self, *args, **kwargs) -> None:
        self.seen_user_message_before_answer = False
        super().__init__(*args, **kwargs)

    def _create_agent(self, workspace, on_event):
        def inspect_state(task: str) -> None:
            state = self.state()
            messages = state["conversations"][0]["messages"]
            self.seen_user_message_before_answer = (
                len(messages) == 1
                and messages[0]["role"] == "user"
                and messages[0]["content"] == task
                and bool(state["running_conversations"])
            )

        return InspectingChatAgent(on_event, inspect_state)


class SlowChatApp(ChatApp):
    def __init__(self, *args, **kwargs) -> None:
        self.started = threading.Event()
        self.finish = threading.Event()
        super().__init__(*args, **kwargs)

    def _create_agent(self, workspace, on_event):
        return SlowChatAgent(on_event, self.started, self.finish)


class ChatTests(unittest.TestCase):
    def test_chat_store_persists_conversation_history(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "conversations.json"
            store = ChatStore(path)
            conversation = store.create_conversation("project-1")

            store.append_pair(
                conversation.id,
                user_message="介绍项目",
                assistant_message="这是一个示例项目。",
                events=["[工具] list_files"],
            )
            reloaded = ChatStore(path)
            loaded = reloaded.get(conversation.id)

            self.assertEqual(loaded.title, "介绍项目")
            self.assertEqual(loaded.messages[0].role, "user")
            self.assertEqual(loaded.messages[1].events, ["[工具] list_files"])

    def test_chat_store_pins_and_deletes_conversations(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "conversations.json"
            store = ChatStore(path)
            first = store.create_conversation("project-1", "普通对话")
            second = store.create_conversation("project-1", "置顶对话")

            store.pin_conversation(first.id, True)
            store.delete_conversation(second.id)
            reloaded = ChatStore(path)
            conversations = reloaded.list_conversations()

            self.assertEqual([conversation.id for conversation in conversations], [first.id])
            self.assertTrue(conversations[0].pinned)

    def test_chat_app_adds_project_and_saves_sent_messages(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            data_dir = Path(temp_dir) / "data"
            project = Path(temp_dir) / "project"
            project.mkdir()
            (project / "main.py").write_text("print('ok')\n", encoding="utf-8")
            app = FakeChatApp(
                data_dir,
                Settings(api_key="test-key", model="test-model"),
            )

            saved_project = app.add_project(str(project))
            conversation = app.create_conversation(str(saved_project["id"]))
            response = app.send_message(str(conversation["id"]), "为我介绍一下这个项目")
            state = app.state()

            self.assertEqual(response["answer"], "已处理：为我介绍一下这个项目")
            self.assertIn("[工具] list_files", response["events"])
            self.assertEqual(len(state["projects"]), 1)
            self.assertEqual(len(state["conversations"]), 1)
            saved_messages = state["conversations"][0]["messages"]
            self.assertEqual(saved_messages[0]["content"], "为我介绍一下这个项目")
            self.assertIn("已处理", saved_messages[1]["content"])

    def test_send_message_persists_user_message_before_agent_runs(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            data_dir = Path(temp_dir) / "data"
            project = Path(temp_dir) / "project"
            project.mkdir()
            app = InspectingChatApp(
                data_dir,
                Settings(api_key="test-key", model="test-model"),
            )
            saved_project = app.add_project(str(project))
            conversation = app.create_conversation(str(saved_project["id"]))

            app.send_message(str(conversation["id"]), "添加性能统计")

            self.assertTrue(app.seen_user_message_before_answer)

    def test_start_message_returns_before_agent_finishes_and_streams_events(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            data_dir = Path(temp_dir) / "data"
            project = Path(temp_dir) / "project"
            project.mkdir()
            app = SlowChatApp(
                data_dir,
                Settings(api_key="test-key", model="test-model"),
            )
            saved_project = app.add_project(str(project))
            conversation = app.create_conversation(str(saved_project["id"]))

            response = app.start_message(str(conversation["id"]), "生成番茄钟")

            self.assertIn("running", response)
            self.assertNotIn("answer", response)
            self.assertTrue(app.started.wait(timeout=2))
            running_state = app.state()["running_conversations"]
            running = running_state[str(conversation["id"])]
            self.assertIn("[状态] 正在检查项目", running["events"])
            messages = app.state()["conversations"][0]["messages"]
            self.assertEqual(messages[-1]["role"], "user")
            self.assertEqual(messages[-1]["content"], "生成番茄钟")

            app.finish.set()
            for _ in range(50):
                state = app.state()
                if not state["running_conversations"]:
                    break
                time.sleep(0.02)
            else:
                self.fail("background chat run did not finish")
            messages = state["conversations"][0]["messages"]
            self.assertEqual(messages[-1]["role"], "assistant")
            self.assertIn("后台完成：生成番茄钟", messages[-1]["content"])
            self.assertIn("[工具] list_files", messages[-1]["events"])

    def test_chat_ui_renders_user_message_and_pending_agent_immediately(self) -> None:
        self.assertIn("appendOptimisticMessages", INDEX_HTML)
        self.assertIn("正在思考", INDEX_HTML)
        self.assertIn("已处理", INDEX_HTML)
        self.assertIn("runningFor", INDEX_HTML)
        self.assertIn("pollState", INDEX_HTML)
        self.assertIn("setInterval", INDEX_HTML)

    def test_chat_ui_renders_assistant_markdown_instead_of_raw_text(self) -> None:
        self.assertIn("renderMarkdown", INDEX_HTML)
        self.assertIn("renderMessageContent(message)", INDEX_HTML)
        self.assertIn("<h${level}>", INDEX_HTML)
        self.assertIn("<ul>", INDEX_HTML)
        self.assertIn("<pre><code>", INDEX_HTML)
        self.assertNotIn("${escapeHtml(message.content)}${events}", INDEX_HTML)

    def test_chat_ui_groups_conversations_under_projects(self) -> None:
        self.assertIn('id="projectTree"', INDEX_HTML)
        self.assertIn("renderProjectTree", INDEX_HTML)
        self.assertIn("data-project-id", INDEX_HTML)
        self.assertIn("data-conversation-id", INDEX_HTML)
        self.assertIn("data-pin-id", INDEX_HTML)
        self.assertIn("data-delete-id", INDEX_HTML)
        self.assertNotIn('id="projectSelect"', INDEX_HTML)
        self.assertNotIn('id="conversationList"', INDEX_HTML)

    def test_chat_ui_has_day_navigation_and_collapsed_finished_events(self) -> None:
        self.assertIn('id="dayNav"', INDEX_HTML)
        self.assertIn("renderDayNav", INDEX_HTML)
        self.assertIn("daysForMessages", INDEX_HTML)
        self.assertIn("day-separator", INDEX_HTML)
        self.assertIn("scrollIntoView", INDEX_HTML)
        self.assertIn("<details class=\"events-details\">", INDEX_HTML)
        self.assertIn("执行过程 ${message.events.length} 条", INDEX_HTML)
        self.assertIn("if (message.pending) return body", INDEX_HTML)

    def test_chat_app_can_pick_and_rename_project_and_conversation(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            data_dir = Path(temp_dir) / "data"
            project = Path(temp_dir) / "project"
            project.mkdir()
            app = FakeChatApp(
                data_dir,
                Settings(api_key="test-key", model="test-model"),
                directory_picker=lambda: str(project),
            )

            saved_project = app.pick_project()
            renamed_project = app.rename_project(str(saved_project["id"]), "番茄钟")
            conversation = app.create_conversation(str(saved_project["id"]))
            renamed_conversation = app.rename_conversation(
                str(conversation["id"]),
                "性能测试",
            )
            state = app.state()

            self.assertEqual(renamed_project["display_name"], "番茄钟")
            self.assertEqual(renamed_conversation["title"], "性能测试")
            self.assertEqual(state["projects"][0]["display_name"], "番茄钟")
            self.assertEqual(state["conversations"][0]["title"], "性能测试")

    def test_chat_app_can_pin_and_delete_conversation(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            data_dir = Path(temp_dir) / "data"
            project = Path(temp_dir) / "project"
            project.mkdir()
            app = FakeChatApp(
                data_dir,
                Settings(api_key="test-key", model="test-model"),
            )
            saved_project = app.add_project(str(project))
            first = app.create_conversation(str(saved_project["id"]), "普通对话")
            second = app.create_conversation(str(saved_project["id"]), "重要对话")

            pinned = app.pin_conversation(str(first["id"]), True)
            deleted = app.delete_conversation(str(second["id"]))
            state = app.state()

            self.assertTrue(pinned["pinned"])
            self.assertEqual(deleted["deleted"], second["id"])
            self.assertEqual(len(state["conversations"]), 1)
            self.assertEqual(state["conversations"][0]["id"], first["id"])
            self.assertTrue(state["conversations"][0]["pinned"])

    def test_chat_app_rejects_deleting_running_conversation(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            data_dir = Path(temp_dir) / "data"
            project = Path(temp_dir) / "project"
            project.mkdir()
            app = SlowChatApp(
                data_dir,
                Settings(api_key="test-key", model="test-model"),
            )
            saved_project = app.add_project(str(project))
            conversation = app.create_conversation(str(saved_project["id"]))

            app.start_message(str(conversation["id"]), "生成番茄钟")
            self.assertTrue(app.started.wait(timeout=2))
            with self.assertRaises(RuntimeError):
                app.delete_conversation(str(conversation["id"]))
            app.finish.set()
            for _ in range(50):
                if not app.state()["running_conversations"]:
                    break
                time.sleep(0.02)
            else:
                self.fail("background chat run did not finish")


if __name__ == "__main__":
    unittest.main()
