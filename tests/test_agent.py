import json
import shutil
import tempfile
import unittest
from pathlib import Path

from coding_agent.agent import Agent, ModelReply, TaskState, TokenUsage, ToolCall
from coding_agent.checkpoint import CheckpointStore
from coding_agent.extensions import BaseExtension, ToolResult
from coding_agent.project_memory import ProjectMemoryStore
from coding_agent.trace import JsonlTrace
from coding_agent.workspace import Workspace


class FakeModel:
    def __init__(self, replies: list[ModelReply | Exception]) -> None:
        self.replies = iter(replies)
        self.received_messages: list[list[dict[str, object]]] = []
        self.received_tools: list[list[dict[str, object]]] = []

    def complete(
        self, messages: list[dict[str, object]], tools: list[dict[str, object]]
    ) -> ModelReply:
        self.received_messages.append([message.copy() for message in messages])
        self.received_tools.append(tools)
        reply = next(self.replies)
        if isinstance(reply, Exception):
            raise reply
        return reply


class InterruptingCheckpointStore(CheckpointStore):
    def __init__(self, path: Path) -> None:
        super().__init__(path)
        self.save_count = 0

    def save(self, data: dict[str, object]) -> None:
        super().save(data)
        self.save_count += 1
        if self.save_count == 1:
            raise KeyboardInterrupt


class RecordingExtension(BaseExtension):
    name = "recording"

    def __init__(self) -> None:
        self.events: list[str] = []

    def on_session_start(self, context) -> None:
        self.events.append(f"start:{context.task}")

    def inject_context(self, context) -> list[dict[str, object]]:
        return [
            {
                "role": "system",
                "content": "<extension_context>project note: beta</extension_context>",
            }
        ]

    def before_llm_call(
        self,
        context,
        *,
        step: int,
        messages: list[dict[str, object]],
        tools: list[dict[str, object]],
    ) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
        self.events.append(f"before_llm:{step}")
        return [
            *messages,
            {"role": "system", "content": "<extension_marker>before llm</extension_marker>"},
        ], tools

    def after_llm_call(self, context, *, step: int, reply) -> None:
        self.events.append(f"after_llm:{step}:{reply.content}")

    def on_session_end(self, context, *, answer, error, state) -> None:
        self.events.append(f"end:{answer}:{error is None}")


class EchoToolExtension(BaseExtension):
    name = "echo-tool"

    def tool_definitions(self, context) -> list[dict[str, object]]:
        return [
            {
                "type": "function",
                "function": {
                    "name": "extension_echo",
                    "description": "Echo text through an extension-owned tool.",
                    "parameters": {
                        "type": "object",
                        "properties": {"text": {"type": "string"}},
                        "required": ["text"],
                    },
                },
            }
        ]

    def execute_tool(self, context, *, step: int, call) -> ToolResult | None:
        if call.name != "extension_echo":
            return None
        return ToolResult(f"extension echo: {call.arguments['text']}")


class BlockingExtension(BaseExtension):
    name = "blocker"

    def before_tool_call(self, context, *, step: int, call):
        if call.name == "read_file":
            return None
        return call


def write_smoke_suite(root: Path, *, passing: bool = True) -> None:
    tests = root / "project_tests"
    tests.mkdir()
    assertion = "self.assertTrue(True)" if passing else "self.fail('not fixed')"
    (tests / "test_smoke.py").write_text(
        "import unittest\n\n"
        "class SmokeTests(unittest.TestCase):\n"
        "    def test_smoke(self):\n"
        f"        {assertion}\n",
        encoding="utf-8",
    )


def full_test_call(call_id: str) -> ToolCall:
    return ToolCall(
        call_id,
        "run_command",
        {
            "argv": [
                "python",
                "-m",
                "unittest",
                "discover",
                "-s",
                "project_tests",
                "-v",
            ]
        },
    )


class AgentTests(unittest.TestCase):
    def test_reports_real_model_token_usage(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            events: list[str] = []
            model = FakeModel(
                [
                    ModelReply(
                        content="完成。",
                        usage=TokenUsage(
                            prompt_tokens=100,
                            completion_tokens=20,
                            total_tokens=120,
                            cache_hit_tokens=40,
                        ),
                    )
                ]
            )

            Agent(
                model,
                Workspace(Path(temp_dir)),
                on_event=events.append,
            ).run("检查项目")

            self.assertIn("[Token] 本次输入 100，输出 20；本次会话累计 120", events)

    def test_follow_up_task_receives_lightweight_session_context_by_default(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            events: list[str] = []
            model = FakeModel(
                [
                    ModelReply(content="已确认这是番茄钟项目。"),
                    ModelReply(content="这个项目是一个番茄钟应用。"),
                ]
            )
            agent = Agent(model, Workspace(Path(temp_dir)), on_event=events.append)

            self.assertFalse(hasattr(agent, "context_manager"))
            self.assertEqual(agent.run("记录：当前项目是番茄钟应用"), "已确认这是番茄钟项目。")
            self.assertEqual(agent.run("为我介绍一下这个项目"), "这个项目是一个番茄钟应用。")

            first_request = json.dumps(model.received_messages[0], ensure_ascii=False)
            second_request = json.dumps(model.received_messages[1], ensure_ascii=False)
            self.assertNotIn("<session_context>", first_request)
            self.assertIn("<session_context>", second_request)
            self.assertIn("当前项目是番茄钟应用", second_request)
            self.assertIn("inspect the current workspace files", second_request)
            self.assertFalse(any(event.startswith("[上下文]") for event in events))

    def test_extension_hooks_can_inject_context_around_model_calls(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            extension = RecordingExtension()
            model = FakeModel([ModelReply(content="完成。")])

            answer = Agent(
                model,
                Workspace(Path(temp_dir)),
                extensions=[extension],
            ).run("检查扩展")

            self.assertEqual(answer, "完成。")
            request = json.dumps(model.received_messages[0], ensure_ascii=False)
            self.assertIn("<extension_context>project note: beta", request)
            self.assertIn("<extension_marker>before llm", request)
            self.assertEqual(
                extension.events,
                [
                    "start:检查扩展",
                    "before_llm:1",
                    "after_llm:1:完成。",
                    "end:完成。:True",
                ],
            )

    def test_extension_can_register_and_execute_a_custom_tool(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            model = FakeModel(
                [
                    ModelReply(
                        content=None,
                        tool_calls=(
                            ToolCall(
                                "echo-1",
                                "extension_echo",
                                {"text": "hello"},
                            ),
                        ),
                    ),
                    ModelReply(content="扩展工具完成。"),
                ]
            )

            answer = Agent(
                model,
                Workspace(Path(temp_dir)),
                extensions=[EchoToolExtension()],
            ).run("调用扩展工具")

            self.assertEqual(answer, "扩展工具完成。")
            tool_names = {tool["function"]["name"] for tool in model.received_tools[0]}
            self.assertIn("extension_echo", tool_names)
            second_request = json.dumps(model.received_messages[1], ensure_ascii=False)
            self.assertIn("extension echo: hello", second_request)

    def test_extension_can_block_a_tool_call(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            (root / "secret.txt").write_text("hidden\n", encoding="utf-8")
            model = FakeModel(
                [
                    ModelReply(
                        content=None,
                        tool_calls=(
                            ToolCall("read-1", "read_file", {"path": "secret.txt"}),
                        ),
                    ),
                    ModelReply(content="读取已被拦截。"),
                ]
            )

            answer = Agent(
                model,
                Workspace(root),
                extensions=[BlockingExtension()],
            ).run("读取文件")

            self.assertEqual(answer, "读取已被拦截。")
            second_request = json.dumps(model.received_messages[1], ensure_ascii=False)
            self.assertIn("tool call blocked by extension", second_request)
            self.assertNotIn("hidden", second_request)

    def test_new_agent_instance_receives_persistent_project_memory(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir) / "project"
            root.mkdir()
            (root / "app.py").write_text("print('timer')\n", encoding="utf-8")
            memory_store = ProjectMemoryStore(Path(temp_dir) / "memory")

            Agent(
                FakeModel([ModelReply(content="番茄钟项目已创建。\npython app.py")]),
                Workspace(root),
                memory_store=memory_store,
            ).run("创建一个番茄钟应用")

            second_model = FakeModel([ModelReply(content="这是番茄钟项目。")])
            Agent(
                second_model,
                Workspace(root),
                memory_store=memory_store,
            ).run("为我介绍一下这个项目")

            second_request = json.dumps(
                second_model.received_messages[0],
                ensure_ascii=False,
            )
            self.assertIn("<project_memory>", second_request)
            self.assertIn("创建一个番茄钟应用", second_request)
            self.assertIn("python app.py", second_request)
            self.assertIn("app.py", second_request)

    def test_bug_report_receives_diagnose_skill(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            trace = JsonlTrace(Path(temp_dir) / "trace.jsonl")
            events: list[str] = []
            model = FakeModel([ModelReply(content="我会先复现再修复。")])

            Agent(
                model,
                Workspace(Path(temp_dir)),
                trace=trace,
                on_event=events.append,
            ).run(
                "运行时报错：Traceback NameError，请修复"
            )

            message_contents = [
                str(message.get("content"))
                for message in model.received_messages[0]
            ]
            trace_events = [
                json.loads(line)
                for line in trace.path.read_text(encoding="utf-8").splitlines()
            ]
            skill_event = next(
                item for item in trace_events if item["event"] == "skills_selected"
            )
            request = "\n".join(message_contents)
            self.assertIn('<skill name="diagnose">', request)
            self.assertIn("Establish a fast feedback loop", request)
            self.assertIn("regression test", request)
            self.assertIn("re-run the original failing path", request.lower())
            self.assertIn("diagnose", skill_event["data"]["skills"])
            self.assertTrue(any(event.startswith("[技能]") for event in events))

    def test_from_scratch_project_is_planned_before_local_tools(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            events: list[str] = []
            model = FakeModel(
                [
                    ModelReply(
                        content=None,
                        tool_calls=(
                            ToolCall(
                                "plan-1",
                                "plan_task",
                                {
                                    "steps": ["设计模块", "实现功能", "运行测试"],
                                    "acceptance": ["应用可以启动", "测试全部通过"],
                                    "test_strategy": "为核心逻辑编写单元测试",
                                },
                            ),
                        ),
                    ),
                    ModelReply(
                        content=None,
                        tool_calls=(ToolCall("review-1", "approve_plan", {}),),
                    ),
                    ModelReply(
                        content=None,
                        tool_calls=(
                            ToolCall(
                                "write-app",
                                "write_file",
                                {"path": "app.py", "content": "VALUE = 1\n"},
                            ),
                            ToolCall(
                                "write-test",
                                "write_file",
                                {
                                    "path": "project_tests/test_app.py",
                                    "content": (
                                        "import unittest\n"
                                        "class AppTests(unittest.TestCase):\n"
                                        "    def test_value(self):\n"
                                        "        self.assertEqual(1, 1)\n"
                                    ),
                                },
                            ),
                            ToolCall(
                                "write-readme",
                                "write_file",
                                {"path": "README.md", "content": "python app.py\n"},
                            ),
                        ),
                    ),
                    ModelReply(
                        content=None,
                        tool_calls=(full_test_call("test-app"),),
                    ),
                    ModelReply(content="计划已确认，尚未修改文件。"),
                ]
            )

            answer = Agent(
                model,
                Workspace(Path(temp_dir)),
                on_event=events.append,
            ).run("请从零创建一个示例应用")

            self.assertEqual(answer, "计划已确认，尚未修改文件。")
            planning_tools = [
                tool["function"]["name"] for tool in model.received_tools[0]
            ]
            self.assertEqual(planning_tools, ["plan_task", "ask_user"])
            self.assertIn(
                "keep the launch entry point thin",
                str(model.received_messages[0][0]["content"]),
            )
            self.assertIn(
                "standard-library unittest",
                str(model.received_messages[0][0]["content"]),
            )
            contract = next(
                str(message["content"])
                for message in model.received_messages[2]
                if "<task_contract>" in str(message.get("content", ""))
            )
            self.assertIn("设计模块 | 实现功能 | 运行测试", contract)
            self.assertIn("为核心逻辑编写单元测试", contract)
            self.assertTrue(any(event.startswith("[计划]") for event in events))

    def test_plan_review_can_restore_an_explicit_interface_requirement(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            model = FakeModel(
                [
                    ModelReply(
                        content=None,
                        tool_calls=(
                            ToolCall(
                                "plan-1",
                                "plan_task",
                                {
                                    "steps": ["实现核心逻辑", "提供 CLI"],
                                    "acceptance": ["命令行可以运行"],
                                    "test_strategy": "测试核心逻辑",
                                },
                            ),
                        ),
                    ),
                    ModelReply(
                        content=None,
                        tool_calls=(
                            ToolCall(
                                "review-1",
                                "reject_plan",
                                {"reason": "必须保留用户要求的桌面 GUI"},
                            ),
                        ),
                    ),
                    ModelReply(
                        content=None,
                        tool_calls=(
                            ToolCall(
                                "revision-1",
                                "plan_task",
                                {
                                    "steps": ["实现核心逻辑", "实现桌面 GUI", "运行测试"],
                                    "acceptance": ["桌面窗口可以启动", "测试通过"],
                                    "test_strategy": "测试核心逻辑并验证 GUI 可启动",
                                },
                            ),
                        ),
                    ),
                    ModelReply(
                        content=None,
                        tool_calls=(
                            ToolCall(
                                "write-app",
                                "write_file",
                                {"path": "app.py", "content": "VALUE = 1\n"},
                            ),
                            ToolCall(
                                "write-test",
                                "write_file",
                                {
                                    "path": "project_tests/test_app.py",
                                    "content": (
                                        "import unittest\n"
                                        "class AppTests(unittest.TestCase):\n"
                                        "    def test_value(self):\n"
                                        "        self.assertTrue(True)\n"
                                    ),
                                },
                            ),
                            ToolCall(
                                "write-readme",
                                "write_file",
                                {"path": "README.md", "content": "python app.py\n"},
                            ),
                        ),
                    ),
                    ModelReply(
                        content=None,
                        tool_calls=(full_test_call("test-app"),),
                    ),
                    ModelReply(content="计划已修正。"),
                ]
            )

            answer = Agent(model, Workspace(Path(temp_dir))).run(
                "请从零创建一个桌面应用"
            )

            self.assertEqual(answer, "计划已修正。")
            contract = next(
                str(message["content"])
                for message in model.received_messages[3]
                if "<task_contract>" in str(message.get("content", ""))
            )
            self.assertIn("实现桌面 GUI", contract)
            self.assertNotIn("提供 CLI", contract)

    def test_planned_project_reviews_missing_deliverables_after_tests_pass(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            write_smoke_suite(root)
            model = FakeModel(
                [
                    ModelReply(
                        content=None,
                        tool_calls=(
                            ToolCall(
                                "plan-1",
                                "plan_task",
                                {
                                    "steps": ["实现应用", "编写 README", "运行测试"],
                                    "acceptance": ["测试通过", "包含使用说明"],
                                    "test_strategy": "运行 unittest",
                                },
                            ),
                        ),
                    ),
                    ModelReply(
                        content=None,
                        tool_calls=(ToolCall("review-1", "approve_plan", {}),),
                    ),
                    ModelReply(
                        content=None,
                        tool_calls=(
                            ToolCall(
                                "write-1",
                                "write_file",
                                {"path": "app.py", "content": "VALUE = 1\n"},
                            ),
                            ToolCall(
                                "write-test",
                                "write_file",
                                {
                                    "path": "project_tests/test_app.py",
                                    "content": (
                                        "import unittest\n"
                                        "class AppTests(unittest.TestCase):\n"
                                        "    def test_value(self):\n"
                                        "        self.assertEqual(1, 1)\n"
                                    ),
                                },
                            ),
                        ),
                    ),
                    ModelReply(content=None, tool_calls=(full_test_call("test-1"),)),
                    ModelReply(
                        content=None,
                        tool_calls=(
                            ToolCall(
                                "write-2",
                                "write_file",
                                {"path": "README.md", "content": "python app.py\n"},
                            ),
                        ),
                    ),
                    ModelReply(content="应用、测试和使用说明均已完成。"),
                ]
            )

            answer = Agent(model, Workspace(root)).run("请从零创建一个 Python 应用")

            self.assertEqual(answer, "应用、测试和使用说明均已完成。")
            self.assertTrue((root / "README.md").exists())
            state_after_tests = str(model.received_messages[4][-1]["content"])
            self.assertIn("Phase: implement", state_after_tests)
            self.assertIn("Missing deliverables: usage documentation", state_after_tests)

    def test_retries_once_when_model_returns_invalid_tool_arguments(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            events: list[str] = []
            model = FakeModel(
                [
                    ValueError("Invalid arguments for tool run_command"),
                    ModelReply(content="检查完成。"),
                ]
            )

            answer = Agent(
                model,
                Workspace(Path(temp_dir)),
                on_event=events.append,
            ).run("检查项目")

            self.assertEqual(answer, "检查完成。")
            self.assertEqual(len(model.received_messages), 2)
            self.assertIn("exactly match its JSON schema", str(model.received_messages[1]))
            self.assertIn("[状态] 模型响应异常，正在重试", events)

    def test_resumes_after_the_last_completed_tool_without_repeating_it(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            checkpoint_path = root / "checkpoint.json"
            interrupted_events: list[str] = []
            first_model = FakeModel(
                [
                    ModelReply(
                        content=None,
                        tool_calls=(
                            ToolCall(
                                "call-1",
                                "write_file",
                                {"path": "first.txt", "content": "first\n"},
                            ),
                            ToolCall(
                                "call-2",
                                "write_file",
                                {"path": "second.txt", "content": "second\n"},
                            ),
                        ),
                    )
                ]
            )
            interrupted_agent = Agent(
                first_model,
                Workspace(root),
                checkpoint_store=InterruptingCheckpointStore(checkpoint_path),
                on_event=interrupted_events.append,
            )

            with self.assertRaises(KeyboardInterrupt):
                interrupted_agent.run("创建两个文件")

            self.assertTrue((root / "first.txt").exists())
            self.assertFalse((root / "second.txt").exists())

            resumed_events: list[str] = []
            resumed_agent = Agent(
                FakeModel([ModelReply(content="两个文件均已创建。")]),
                Workspace(root),
                checkpoint_store=CheckpointStore(checkpoint_path),
                on_event=resumed_events.append,
            )
            answer = resumed_agent.resume()

            self.assertEqual(answer, "两个文件均已创建。")
            self.assertEqual(
                [event for event in interrupted_events if event.startswith("[工具]")],
                ["[工具] write_file"],
            )
            self.assertEqual(
                [event for event in resumed_events if event.startswith("[工具]")],
                ["[工具] write_file"],
            )
            self.assertEqual((root / "first.txt").read_text(), "first\n")
            self.assertEqual((root / "second.txt").read_text(), "second\n")
            self.assertFalse(checkpoint_path.exists())
    def test_ambiguous_task_pauses_for_clarification_without_changes(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            model = FakeModel(
                [
                    ModelReply(
                        content=None,
                        tool_calls=(
                            ToolCall(
                                "call-1",
                                "ask_user",
                                {"question": "你希望优化哪一方面？"},
                            ),
                        ),
                    ),
                    ModelReply(content="已明确目标，准备处理性能问题。"),
                ]
            )
            agent = Agent(model, Workspace(root))

            question = agent.run("帮我优化这个项目")

            self.assertEqual(question, "你希望优化哪一方面？")
            self.assertTrue(agent.awaiting_clarification)
            self.assertEqual(list(root.iterdir()), [])
            gate_tools = {
                str(definition["function"]["name"])
                for definition in model.received_tools[0]
            }
            self.assertEqual(gate_tools, {"ask_user", "proceed_task"})

            answer = agent.run("优化列表加载速度，保持现有接口不变")

            self.assertEqual(answer, "已明确目标，准备处理性能问题。")
            self.assertFalse(agent.awaiting_clarification)
            continued_task = next(
                str(message.get("content"))
                for message in model.received_messages[1]
                if message.get("role") == "user"
            )
            self.assertIn("帮我优化这个项目", continued_task)
            self.assertIn("优化列表加载速度", continued_task)

    def test_resumes_a_pending_clarification_after_restart(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            store = CheckpointStore(root / "checkpoint.json")
            first_agent = Agent(
                FakeModel(
                    [
                        ModelReply(
                            content=None,
                            tool_calls=(
                                ToolCall(
                                    "call-1",
                                    "ask_user",
                                    {"question": "需要优化速度还是可读性？"},
                                ),
                            ),
                        )
                    ]
                ),
                Workspace(root),
                checkpoint_store=store,
            )
            self.assertEqual(
                first_agent.run("优化项目"),
                "需要优化速度还是可读性？",
            )

            resumed_model = FakeModel([ModelReply(content="将优化运行速度。")])
            resumed_agent = Agent(
                resumed_model,
                Workspace(root),
                checkpoint_store=store,
            )

            self.assertEqual(resumed_agent.resume(), "需要优化速度还是可读性？")
            self.assertTrue(resumed_agent.awaiting_clarification)
            self.assertEqual(resumed_agent.run("运行速度"), "将优化运行速度。")
            continued_task = str(resumed_model.received_messages[0][1]["content"])
            self.assertIn("优化项目", continued_task)
            self.assertIn("运行速度", continued_task)
            self.assertFalse(store.path.exists())
    def test_allows_a_final_answer_after_the_last_tool_step(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            model = FakeModel(
                [
                    ModelReply(
                        content=None,
                        tool_calls=(ToolCall("call-1", "list_files", {}),),
                    ),
                    ModelReply(content="检查完成。"),
                ]
            )

            answer = Agent(model, Workspace(Path(temp_dir)), max_steps=1).run(
                "检查项目"
            )

            self.assertEqual(answer, "检查完成。")
            self.assertEqual(model.received_tools[1], [])

    def test_retries_when_final_answer_contains_tool_protocol(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            model = FakeModel(
                [
                    ModelReply(
                        content=None,
                        tool_calls=(ToolCall("call-1", "list_files", {}),),
                    ),
                    ModelReply(
                        content=(
                            "检查完成。\n<｜DSML｜tool_calls>"
                            "<｜DSML｜invoke name=run_command>"
                        )
                    ),
                    ModelReply(content="检查完成，未修改文件。"),
                ]
            )

            answer = Agent(model, Workspace(Path(temp_dir)), max_steps=1).run(
                "检查项目"
            )

            self.assertEqual(answer, "检查完成，未修改文件。")
            self.assertEqual(model.received_tools[1], [])
            self.assertEqual(model.received_tools[2], [])

    def test_model_can_observe_a_tool_result_before_answering(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            (root / "README.md").write_text("# Demo\n", encoding="utf-8")
            model = FakeModel(
                [
                    ModelReply(
                        content=None,
                        tool_calls=(
                            ToolCall("call-1", "list_files", {"path": "."}),
                        ),
                    ),
                    ModelReply(content="项目包含 README.md。"),
                ]
            )

            answer = Agent(model, Workspace(root)).run("解释项目结构")

            self.assertEqual(answer, "项目包含 README.md。")
            second_request = model.received_messages[1]
            self.assertTrue(
                any(
                    message.get("role") == "tool"
                    and "README.md" in str(message.get("content"))
                    for message in second_request
                )
            )

    def test_model_receives_updated_task_state_after_file_change(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            model = FakeModel(
                [
                    ModelReply(
                        content=None,
                        tool_calls=(
                            ToolCall(
                                "call-1",
                                "write_file",
                                {"path": "result.txt", "content": "done\n"},
                            ),
                        ),
                    ),
                    ModelReply(content="文件已创建。"),
                ]
            )

            Agent(model, Workspace(Path(temp_dir)), max_steps=2).run("创建结果文件")

            second_request = model.received_messages[1]
            state_messages = [
                str(message.get("content"))
                for message in second_request
                if message.get("role") == "system"
                and "<task_state>" in str(message.get("content"))
            ]
            self.assertEqual(len(state_messages), 1)
            self.assertIn("Phase: verify", state_messages[0])
            self.assertIn("Modified files: result.txt", state_messages[0])
            self.assertIn("Remaining action rounds: 1", state_messages[0])
            available = {
                definition["function"]["name"]
                for definition in model.received_tools[1]
            }
            self.assertEqual(
                available,
                {"write_file", "apply_patch", "run_command", "ask_user"},
            )

    def test_plain_command_does_not_count_as_full_verification(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            (root / "check.py").write_text("print('ok')\n", encoding="utf-8")
            model = FakeModel(
                [
                    ModelReply(
                        content=None,
                        tool_calls=(
                            ToolCall(
                                "call-1",
                                "run_command",
                                {"argv": ["python", "check.py"]},
                            ),
                        ),
                    ),
                    ModelReply(content="验证通过。"),
                ]
            )

            Agent(model, Workspace(root), max_steps=2).run("验证项目")

            state = str(model.received_messages[1][-1]["content"])
            self.assertIn("Phase: inspect", state)
            self.assertIn("Latest command: passed (exit 0)", state)
            self.assertNotIn("Return a final answer now", state)

    def test_explicit_unittest_module_counts_as_full_verification(self) -> None:
        call = ToolCall(
            "call-1",
            "run_command",
            {"argv": ["python", "-m", "unittest", "-v", "test_app.py"]},
        )
        state = TaskState("创建 Python 应用", changes_pending_verification=True)

        state.update(call, "Exit code: 0\nSTDERR:\nRan 1 test\nOK", success=True)

        self.assertTrue(state.full_tests_passed)
        self.assertTrue(state.ready_to_finalize)

    def test_creation_stays_in_implementation_until_deliverables_exist(self) -> None:
        state = TaskState(
            "创建 Python 桌面应用",
            creation_task=True,
            plan=["实现应用", "编写测试", "编写 README", "运行测试"],
        )

        state.update(
            ToolCall(
                "call-1",
                "write_file",
                {"path": "timer.py", "content": "class Timer: pass\n"},
            ),
            "Wrote timer.py",
            success=True,
        )

        self.assertEqual(state.phase, "implement")
        self.assertIn("entry point", state.missing_deliverables)
        self.assertIn("automated tests", state.missing_deliverables)
        self.assertIn("usage documentation", state.missing_deliverables)

        for call_id, path in (
            ("call-2", "main.py"),
            ("call-3", "tests/test_timer.py"),
            ("call-4", "README.md"),
        ):
            state.update(
                ToolCall(call_id, "write_file", {"path": path, "content": "x\n"}),
                f"Wrote {path}",
                success=True,
            )

        self.assertEqual(state.missing_deliverables, [])
        self.assertEqual(state.phase, "verify")

    def test_zero_discovered_tests_do_not_count_as_verification(self) -> None:
        call = ToolCall(
            "call-1",
            "run_command",
            {"argv": ["python", "-m", "unittest", "discover"]},
        )
        state = TaskState("创建 Python 应用", changes_pending_verification=True)

        state.update(call, "Exit code: 0\nSTDERR:\nRan 0 tests\nOK", success=True)

        self.assertFalse(state.full_tests_passed)
        self.assertFalse(state.ready_to_finalize)
        self.assertEqual(state.phase, "verify")
        self.assertIn("no tests", state.last_error.lower())

    def test_finalizes_without_tools_after_changed_files_pass_full_tests(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            write_smoke_suite(root)
            model = FakeModel(
                [
                    ModelReply(
                        content=None,
                        tool_calls=(
                            ToolCall(
                                "call-1",
                                "write_file",
                                {"path": "result.txt", "content": "done\n"},
                            ),
                        ),
                    ),
                    ModelReply(
                        content=None,
                        tool_calls=(full_test_call("call-2"),),
                    ),
                    ModelReply(content="修改完成，完整测试通过。"),
                ]
            )

            answer = Agent(model, Workspace(root)).run("修改并测试项目")

            self.assertEqual(answer, "修改完成，完整测试通过。")
            self.assertEqual(model.received_tools[2], [])
            self.assertTrue(
                any(
                    "Full test suite passed" in str(message.get("content"))
                    for message in model.received_messages[2]
                )
            )
            initial_prompt = str(model.received_messages[0][0]["content"])
            final_request = "\n".join(
                str(message.get("content")) for message in model.received_messages[2]
            )
            self.assertIn("create or update necessary automated tests", initial_prompt)
            self.assertIn("standard-library unittest", initial_prompt)
            self.assertIn("Do not read a file back immediately", initial_prompt)
            self.assertIn("Batch independent tool calls", initial_prompt)
            self.assertIn("exact usage or launch instructions", final_request)

    def test_passing_tests_before_a_change_do_not_trigger_finalization(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            write_smoke_suite(root)
            model = FakeModel(
                [
                    ModelReply(
                        content=None,
                        tool_calls=(full_test_call("call-1"),),
                    ),
                    ModelReply(
                        content=None,
                        tool_calls=(
                            ToolCall(
                                "call-2",
                                "write_file",
                                {"path": "result.txt", "content": "changed\n"},
                            ),
                        ),
                    ),
                    ModelReply(content="修改完成，尚未验证。"),
                ]
            )

            Agent(model, Workspace(root)).run("先检查再修改")

            self.assertNotEqual(model.received_tools[1], [])
            self.assertNotEqual(model.received_tools[2], [])

    def test_failed_full_tests_do_not_trigger_finalization(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            write_smoke_suite(root, passing=False)
            model = FakeModel(
                [
                    ModelReply(
                        content=None,
                        tool_calls=(
                            ToolCall(
                                "call-1",
                                "write_file",
                                {"path": "result.txt", "content": "changed\n"},
                            ),
                        ),
                    ),
                    ModelReply(
                        content=None,
                        tool_calls=(full_test_call("call-2"),),
                    ),
                    ModelReply(content="测试仍然失败。"),
                ]
            )

            answer = Agent(model, Workspace(root)).run("修改并测试项目")

            self.assertEqual(answer, "测试仍然失败。")
            self.assertNotEqual(model.received_tools[2], [])
            repair_request = json.dumps(model.received_messages[2], ensure_ascii=False)
            self.assertIn("Phase: repair", repair_request)
            self.assertIn("reproduce the symptom", repair_request)
            self.assertIn("test one focused hypothesis", repair_request)
            self.assertIn("rerun the failing path", repair_request)

    def test_does_not_accept_unverified_code_changes_as_complete(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            write_smoke_suite(root)
            model = FakeModel(
                [
                    ModelReply(
                        content=None,
                        tool_calls=(
                            ToolCall(
                                "call-1",
                                "write_file",
                                {"path": "app.py", "content": "VALUE = 1\n"},
                            ),
                        ),
                    ),
                    ModelReply(content="代码已经完成。"),
                    ModelReply(
                        content=None,
                        tool_calls=(full_test_call("call-2"),),
                    ),
                    ModelReply(content="代码完成，完整测试通过。"),
                ]
            )

            answer = Agent(model, Workspace(root)).run("创建 Python 模块")

            self.assertEqual(answer, "代码完成，完整测试通过。")
            self.assertNotEqual(model.received_tools[2], [])
            self.assertEqual(model.received_tools[3], [])

    def test_uses_runtime_summary_when_verified_final_answer_stays_invalid(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            write_smoke_suite(root)
            invalid_answer = ModelReply(
                content="<｜DSML｜tool_calls><｜DSML｜invoke name=run_command>"
            )
            model = FakeModel(
                [
                    ModelReply(
                        content=None,
                        tool_calls=(
                            ToolCall(
                                "call-1",
                                "write_file",
                                {"path": "result.txt", "content": "done\n"},
                            ),
                        ),
                    ),
                    ModelReply(
                        content=None,
                        tool_calls=(full_test_call("call-2"),),
                    ),
                    invalid_answer,
                    invalid_answer,
                ]
            )

            answer = Agent(model, Workspace(root)).run("修改并测试项目")

            self.assertIn("完整测试已通过", answer)
            self.assertIn("result.txt", answer)
            self.assertNotIn("DSML", answer)

    def test_records_each_agent_step_in_trace(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            (root / "README.md").write_text("# Demo\n", encoding="utf-8")
            trace_path = root / "trace.jsonl"
            model = FakeModel(
                [
                    ModelReply(
                        content=None,
                        tool_calls=(ToolCall("call-1", "list_files", {}),),
                    ),
                    ModelReply(content="完成。"),
                ]
            )

            Agent(
                model,
                Workspace(root),
                trace=JsonlTrace(trace_path),
            ).run("查看项目")

            events = {
                json.loads(line)["event"]
                for line in trace_path.read_text(encoding="utf-8").splitlines()
            }
            self.assertTrue(
                {
                    "task_start",
                    "model_request",
                    "model_reply",
                    "tool_start",
                    "tool_result",
                    "task_complete",
                }.issubset(events)
            )

    def test_reports_live_progress_for_inspection_editing_and_testing(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            write_smoke_suite(root)
            events: list[str] = []
            model = FakeModel(
                [
                    ModelReply(
                        content=None,
                        tool_calls=(ToolCall("call-1", "list_files", {}),),
                    ),
                    ModelReply(
                        content=None,
                        tool_calls=(
                            ToolCall(
                                "call-2",
                                "write_file",
                                {"path": "app.py", "content": "VALUE = 1\n"},
                            ),
                        ),
                    ),
                    ModelReply(
                        content=None,
                        tool_calls=(full_test_call("call-3"),),
                    ),
                    ModelReply(content="完成。"),
                ]
            )

            Agent(model, Workspace(root), on_event=events.append).run("创建模块")

            statuses = [event for event in events if event.startswith("[状态]")]
            self.assertIn("[状态] 正在分析任务", statuses)
            self.assertIn("[状态] 正在检查项目", statuses)
            self.assertIn("[状态] 正在修改 app.py", statuses)
            self.assertIn("[状态] 正在运行测试", statuses)
            self.assertIn("[状态] 测试通过，正在整理结果", statuses)

    def test_agent_can_modify_fixture_and_run_its_tests(self) -> None:
        fixture = Path(__file__).parent / "fixtures" / "todo_project"
        with tempfile.TemporaryDirectory() as temp_dir:
            project = Path(temp_dir) / "todo_project"
            shutil.copytree(fixture, project)
            new_function = """    return updated


def delete_task(tasks: list[Task], task_id: int) -> list[Task]:
    remaining = [task for task in tasks if task["id"] != task_id]
    if len(remaining) == len(tasks):
        raise ValueError(f"Task does not exist: {task_id}")
    return remaining
"""
            new_test = """import unittest

from todo_app.core import delete_task


class DeleteTaskTests(unittest.TestCase):
    def test_deletes_existing_task(self) -> None:
        tasks = [{"id": 1, "title": "Demo", "done": False}]
        self.assertEqual(delete_task(tasks, 1), [])
"""
            model = FakeModel(
                [
                    ModelReply(
                        content=None,
                        tool_calls=(
                            ToolCall(
                                "call-1",
                                "search_text",
                                {"query": "def complete_task", "path": "."},
                            ),
                        ),
                    ),
                    ModelReply(
                        content=None,
                        tool_calls=(
                            ToolCall(
                                "call-2",
                                "apply_patch",
                                {
                                    "path": "todo_app/core.py",
                                    "old_text": "    return updated\n",
                                    "new_text": new_function,
                                },
                            ),
                        ),
                    ),
                    ModelReply(
                        content=None,
                        tool_calls=(
                            ToolCall(
                                "call-3",
                                "write_file",
                                {
                                    "path": "project_tests/test_delete.py",
                                    "content": new_test,
                                },
                            ),
                        ),
                    ),
                    ModelReply(
                        content=None,
                        tool_calls=(
                            ToolCall(
                                "call-4",
                                "run_command",
                                {
                                    "argv": [
                                        "python",
                                        "-m",
                                        "unittest",
                                        "discover",
                                        "-s",
                                        "project_tests",
                                        "-v",
                                    ]
                                },
                            ),
                        ),
                    ),
                    ModelReply(content="删除功能已添加，全部测试通过。"),
                ]
            )

            answer = Agent(model, Workspace(project)).run("增加删除任务功能")

            self.assertEqual(answer, "删除功能已添加，全部测试通过。")
            self.assertIn(
                "def delete_task",
                (project / "todo_app" / "core.py").read_text(encoding="utf-8"),
            )
            self.assertTrue(
                any(
                    message.get("role") == "tool"
                    and "Exit code: 0" in str(message.get("content"))
                    for message in model.received_messages[-1]
                )
            )


if __name__ == "__main__":
    unittest.main()
