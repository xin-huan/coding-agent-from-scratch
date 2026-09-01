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


class MutatingChatAgent:
    def __init__(self, workspace, on_event) -> None:
        self.workspace = workspace
        self.on_event = on_event

    def run(self, task: str) -> str:
        self.on_event("[状态] 正在修改文件")
        (self.workspace.root / "main.py").write_text(
            f"print({task!r})\n",
            encoding="utf-8",
        )
        return "修改完成"


class FakeChatApp(ChatApp):
    def _create_agent(self, workspace, on_event):
        return FakeChatAgent(on_event)


class MutatingChatApp(ChatApp):
    def _create_agent(self, workspace, on_event):
        return MutatingChatAgent(workspace, on_event)


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

    def test_chat_store_preserves_tree_branches_from_history_nodes(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "conversations.json"
            store = ChatStore(path)
            conversation = store.create_conversation("project-1")

            first_branch = store.append_pair(
                conversation.id,
                user_message="方案 A",
                assistant_message="A 结果",
                events=[],
            )
            branch_point = first_branch.current_message_id
            store.append_pair(
                conversation.id,
                user_message="继续 A",
                assistant_message="A 后续",
                events=[],
            )
            store.checkout_message(conversation.id, branch_point)
            store.append_pair(
                conversation.id,
                user_message="尝试 B",
                assistant_message="B 结果",
                events=[],
            )

            loaded = ChatStore(path).get(conversation.id)
            current_contents = [message.content for message in loaded.current_messages()]
            all_contents = [message.content for message in loaded.messages]

            self.assertEqual(current_contents, ["方案 A", "A 结果", "尝试 B", "B 结果"])
            self.assertIn("继续 A", all_contents)
            self.assertEqual(len(loaded.messages), 6)
            children = [
                message
                for message in loaded.messages
                if message.parent_id == branch_point
            ]
            self.assertEqual({message.content for message in children}, {"继续 A", "尝试 B"})

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
            self.assertIn("message_tree", state["conversations"][0])
            self.assertTrue(saved_messages[0]["id"])
            self.assertEqual(saved_messages[1]["parent_id"], saved_messages[0]["id"])

    def test_chat_app_can_checkout_history_node_and_continue_new_branch(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            data_dir = Path(temp_dir) / "data"
            project = Path(temp_dir) / "project"
            project.mkdir()
            app = FakeChatApp(
                data_dir,
                Settings(api_key="test-key", model="test-model"),
            )
            saved_project = app.add_project(str(project))
            conversation = app.create_conversation(str(saved_project["id"]))

            first = app.send_message(str(conversation["id"]), "方案 A")["conversation"]
            branch_point = first["current_message_id"]
            app.send_message(str(conversation["id"]), "继续 A")

            self.assertIn(str(conversation["id"]), app._agents)
            checked_out = app.checkout_message(str(conversation["id"]), branch_point)
            self.assertNotIn(str(conversation["id"]), app._agents)
            self.assertEqual([item["content"] for item in checked_out["messages"]], ["方案 A", "已处理：方案 A"])

            branched = app.send_message(str(conversation["id"]), "尝试 B")["conversation"]
            visible = [item["content"] for item in branched["messages"]]
            full_tree = [item["content"] for item in branched["message_tree"]]

            self.assertEqual(visible, ["方案 A", "已处理：方案 A", "尝试 B", "已处理：尝试 B"])
            self.assertIn("继续 A", full_tree)
            self.assertEqual(len(branched["message_tree"]), 6)

    def test_chat_app_binds_workspace_snapshot_to_completed_agent_turn(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            data_dir = Path(temp_dir) / "data"
            project = Path(temp_dir) / "project"
            project.mkdir()
            (project / "main.py").write_text("print('before')\n", encoding="utf-8")
            app = MutatingChatApp(
                data_dir,
                Settings(api_key="test-key", model="test-model"),
            )
            saved_project = app.add_project(str(project))
            conversation = app.create_conversation(str(saved_project["id"]))

            response = app.send_message(str(conversation["id"]), "after")
            messages = response["conversation"]["messages"]
            snapshot_id = messages[-1]["workspace_snapshot_id"]

            self.assertTrue(snapshot_id)
            self.assertTrue((data_dir / "workspace-snapshots" / f"{snapshot_id}.json").exists())

            (project / "main.py").write_text("print('current work')\n", encoding="utf-8")
            restored = app.restore_workspace_snapshot(str(conversation["id"]), str(snapshot_id))

            self.assertEqual((project / "main.py").read_text(encoding="utf-8"), "print('after')\n")
            self.assertEqual(restored["restored_files"], ["main.py"])
            self.assertTrue(restored["backup_id"])
            self.assertTrue((data_dir / "workspace-snapshots" / f"{restored['backup_id']}.json").exists())

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
        self.assertIn("data-project-delete-id", INDEX_HTML)
        self.assertIn("data-project-drag-id", INDEX_HTML)
        self.assertIn("/api/projects/delete", INDEX_HTML)
        self.assertIn("/api/projects/reorder", INDEX_HTML)
        self.assertIn("moveProjectTo", INDEX_HTML)
        self.assertIn("dragstart", INDEX_HTML)
        self.assertIn("dragover", INDEX_HTML)
        self.assertIn("drop", INDEX_HTML)
        self.assertIn("projectsInitialized", INDEX_HTML)
        self.assertIn("focusProject(projectId)", INDEX_HTML)
        self.assertIn("data-conversation-id", INDEX_HTML)
        self.assertIn("data-pin-id", INDEX_HTML)
        self.assertIn("data-delete-id", INDEX_HTML)
        self.assertNotIn('id="projectSelect"', INDEX_HTML)
        self.assertNotIn('id="conversationList"', INDEX_HTML)

    def test_chat_ui_has_day_navigation_and_collapsed_finished_events(self) -> None:
        self.assertIn('id="dayNav"', INDEX_HTML)
        self.assertIn("renderDayNav", INDEX_HTML)
        self.assertIn("renderSessionTree", INDEX_HTML)
        self.assertIn('message.role === "user"', INDEX_HTML)
        self.assertIn("resumeIdForUser", INDEX_HTML)
        self.assertIn("const allTurns = nodes.filter", INDEX_HTML)
        self.assertIn("visibleDepthForTurn", INDEX_HTML)
        self.assertIn("--tree-indent", INDEX_HTML)
        self.assertIn("turn-menu", INDEX_HTML)
        self.assertNotIn("renderTurnDetailCard", INDEX_HTML)
        self.assertNotIn("turn-detail-card", INDEX_HTML)
        self.assertNotIn("padding-left:${8 + depth * 12}px", INDEX_HTML)
        self.assertIn("data-message-checkout", INDEX_HTML)
        self.assertIn("data-message-scroll", INDEX_HTML)
        self.assertIn("data-snapshot-id", INDEX_HTML)
        self.assertIn("data-turn-select", INDEX_HTML)
        self.assertIn("data-turn-continue", INDEX_HTML)
        self.assertIn("data-turn-restore", INDEX_HTML)
        self.assertIn("/api/conversations/checkout", INDEX_HTML)
        self.assertIn("/api/conversations/restore-snapshot", INDEX_HTML)
        self.assertIn("daysForMessages", INDEX_HTML)
        self.assertIn("day-separator", INDEX_HTML)
        self.assertIn("scrollIntoView", INDEX_HTML)
        self.assertIn("<details class=\"events-details\">", INDEX_HTML)
        self.assertIn("执行过程 ${message.events.length} 条", INDEX_HTML)
        self.assertIn("if (message.pending) return body", INDEX_HTML)
        self.assertIn("contextmenu", INDEX_HTML)
        self.assertIn("openTurnMenu(nodeButton", INDEX_HTML)
        self.assertIn("restoreSnapshotForTurn", INDEX_HTML)

    def test_chat_ui_scrolls_checkout_to_user_turn_top(self) -> None:
        self.assertIn("scrollToMessage(scrollMessageId || messageId)", INDEX_HTML)
        self.assertIn("function messageElementId", INDEX_HTML)
        self.assertIn('id="${messageElementId(message.id)}"', INDEX_HTML)
        self.assertIn('data-message-scroll="${escapeHtml(userId)}"', INDEX_HTML)
        self.assertIn('scrollToMessage(nodeButton.dataset.turnUser || "")', INDEX_HTML)

    def test_chat_ui_captures_real_parent_before_optimistic_message(self) -> None:
        capture = INDEX_HTML.index('const parentMessageId = currentConversation()?.current_message_id || "";')
        optimistic = INDEX_HTML.index("appendOptimisticMessages(state.conversationId, content);")
        request = INDEX_HTML.index("parent_message_id: parentMessageId")

        self.assertLess(capture, optimistic)
        self.assertLess(optimistic, request)
        self.assertNotIn('parent_message_id: currentConversation()?.current_message_id || ""', INDEX_HTML)

    def test_chat_ui_supports_resizable_columns(self) -> None:
        self.assertIn("--sidebar-width", INDEX_HTML)
        self.assertIn("--nav-width", INDEX_HTML)
        self.assertIn('id="appResizer"', INDEX_HTML)
        self.assertIn('id="navResizer"', INDEX_HTML)
        self.assertIn("setupResizers", INDEX_HTML)
        self.assertIn("pointerdown", INDEX_HTML)
        self.assertIn("localStorage.setItem", INDEX_HTML)

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

    def test_chat_ui_has_windows_directory_picker_fallback(self) -> None:
        self.assertIn("选择项目失败", INDEX_HTML)
        self.assertIn("选择中...", INDEX_HTML)
        with open("src/coding_agent/chat.py", encoding="utf-8") as handle:
            source = handle.read()
        self.assertIn("BrowseForFolder", source)
        self.assertIn("filedialog.askdirectory", source)
        self.assertIn("FolderBrowserDialog", source)
        self.assertIn("-STA", source)
        self.assertNotIn("CREATE_NO_WINDOW", source)

    def test_chat_app_can_reorder_projects(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            data_dir = Path(temp_dir) / "data"
            first_root = Path(temp_dir) / "first"
            second_root = Path(temp_dir) / "second"
            first_root.mkdir()
            second_root.mkdir()
            app = FakeChatApp(
                data_dir,
                Settings(api_key="test-key", model="test-model"),
            )
            first = app.add_project(str(first_root))
            second = app.add_project(str(second_root))

            result = app.reorder_projects([str(second["id"]), str(first["id"])])
            state = app.state()

            self.assertEqual([project["id"] for project in result["projects"]], [second["id"], first["id"]])
            self.assertEqual([project["id"] for project in state["projects"]], [second["id"], first["id"]])

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

    def test_chat_app_can_delete_project_without_deleting_workspace_files(self) -> None:
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
            conversation = app.create_conversation(str(saved_project["id"]), "要删除的对话")

            deleted = app.delete_project(str(saved_project["id"]))
            state = app.state()

            self.assertEqual(deleted["deleted"], saved_project["id"])
            self.assertEqual(deleted["deleted_conversations"], [conversation["id"]])
            self.assertEqual(state["projects"], [])
            self.assertEqual(state["conversations"], [])
            self.assertTrue((project / "main.py").exists())

    def test_chat_app_rejects_deleting_running_project(self) -> None:
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
                app.delete_project(str(saved_project["id"]))
            app.finish.set()
            for _ in range(50):
                if not app.state()["running_conversations"]:
                    break
                time.sleep(0.02)
            else:
                self.fail("background chat run did not finish")

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
