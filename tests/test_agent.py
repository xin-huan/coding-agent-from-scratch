import json
import shutil
import tempfile
import unittest
from pathlib import Path

from coding_agent.agent import Agent, ModelReply, ToolCall
from coding_agent.trace import JsonlTrace
from coding_agent.workspace import Workspace


class FakeModel:
    def __init__(self, replies: list[ModelReply]) -> None:
        self.replies = iter(replies)
        self.received_messages: list[list[dict[str, object]]] = []

    def complete(
        self, messages: list[dict[str, object]], tools: list[dict[str, object]]
    ) -> ModelReply:
        self.received_messages.append([message.copy() for message in messages])
        return next(self.replies)


class AgentTests(unittest.TestCase):
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
