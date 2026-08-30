import json
import shutil
import tempfile
import unittest
from pathlib import Path

from coding_agent.agent import Agent, ModelReply, ToolCall
from coding_agent.checkpoint import CheckpointStore
from coding_agent.trace import JsonlTrace
from coding_agent.workspace import Workspace


class FakeModel:
    def __init__(self, replies: list[ModelReply]) -> None:
        self.replies = iter(replies)
        self.received_messages: list[list[dict[str, object]]] = []
        self.received_tools: list[list[dict[str, object]]] = []

    def complete(
        self, messages: list[dict[str, object]], tools: list[dict[str, object]]
    ) -> ModelReply:
        self.received_messages.append([message.copy() for message in messages])
        self.received_tools.append(tools)
        return next(self.replies)


class InterruptingCheckpointStore(CheckpointStore):
    def __init__(self, path: Path) -> None:
        super().__init__(path)
        self.save_count = 0

    def save(self, data: dict[str, object]) -> None:
        super().save(data)
        self.save_count += 1
        if self.save_count == 1:
            raise KeyboardInterrupt


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
            self.assertEqual(interrupted_events, ["[工具] write_file"])
            self.assertEqual(resumed_events, ["[工具] write_file"])
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
            continued_task = str(model.received_messages[1][1]["content"])
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
