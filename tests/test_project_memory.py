import json
import tempfile
import unittest
from pathlib import Path

from coding_agent.project_memory import (
    ProjectMemoryStore,
    extract_launch_commands,
    extract_user_decisions,
)
from coding_agent.workspace import Workspace


class ProjectMemoryTests(unittest.TestCase):
    def test_persists_project_memory_without_secret_files(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir) / "project"
            root.mkdir()
            (root / "app.py").write_text("print('ok')\n", encoding="utf-8")
            (root / ".env").write_text("DEEPSEEK_API_KEY=secret\n", encoding="utf-8")
            (root / ".git").mkdir()
            (root / ".git" / "HEAD").write_text("main\n", encoding="utf-8")
            store = ProjectMemoryStore(Path(temp_dir) / "memory")

            memory = store.update_after_task(
                Workspace(root),
                task="Original task:\n创建番茄钟\n\nUser clarification:\n使用 Tkinter，保持标准库实现",
                answer="运行方式：\npython app.py\n",
                modified_files=["app.py"],
                latest_command="passed (exit 0)",
            )
            reloaded = store.load(Workspace(root))

            self.assertEqual(reloaded.project_id, memory.project_id)
            self.assertIn("app.py", reloaded.structure)
            self.assertNotIn(".env", reloaded.structure)
            self.assertNotIn(".git/HEAD", reloaded.structure)
            self.assertIn("使用 Tkinter，保持标准库实现", reloaded.user_decisions)
            self.assertIn("python app.py", reloaded.launch_commands)
            self.assertEqual(reloaded.tasks[-1].modified_files, ["app.py"])
            self.assertEqual(reloaded.display_name, "project")

            raw = json.loads(store.path_for(Workspace(root)).read_text(encoding="utf-8"))
            self.assertEqual(raw["version"], 1)

    def test_renames_project_memory(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir) / "project"
            root.mkdir()
            workspace = Workspace(root)
            store = ProjectMemoryStore(Path(temp_dir) / "memory")
            memory = store.update_after_task(
                workspace,
                task="创建项目",
                answer="完成。",
                modified_files=[],
                latest_command="not run",
            )

            renamed = store.rename_project(memory.project_id, "番茄钟")

            self.assertEqual(renamed.display_name, "番茄钟")
            self.assertEqual(store.load(workspace).display_name, "番茄钟")

    def test_reorders_project_memory(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            first_root = Path(temp_dir) / "first"
            second_root = Path(temp_dir) / "second"
            first_root.mkdir()
            second_root.mkdir()
            store = ProjectMemoryStore(Path(temp_dir) / "memory")
            first = store.update_after_task(
                Workspace(first_root),
                task="创建第一个项目",
                answer="完成。",
                modified_files=[],
                latest_command="not run",
            )
            second = store.update_after_task(
                Workspace(second_root),
                task="创建第二个项目",
                answer="完成。",
                modified_files=[],
                latest_command="not run",
            )

            reordered = store.reorder_projects([second.project_id, first.project_id])

            self.assertEqual([project.project_id for project in reordered], [second.project_id, first.project_id])
            self.assertEqual(store.load(Workspace(second_root)).sort_order, 0)
            self.assertEqual(store.load(Workspace(first_root)).sort_order, 1)

    def test_deletes_project_memory_without_deleting_workspace(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir) / "project"
            root.mkdir()
            (root / "main.py").write_text("print('ok')\n", encoding="utf-8")
            workspace = Workspace(root)
            store = ProjectMemoryStore(Path(temp_dir) / "memory")
            memory = store.update_after_task(
                workspace,
                task="创建项目",
                answer="完成。",
                modified_files=["main.py"],
                latest_command="not run",
            )

            store.delete_project(memory.project_id)

            self.assertTrue(root.exists())
            self.assertTrue((root / "main.py").exists())
            self.assertEqual(store.list_projects(), [])

    def test_builds_project_memory_message_for_future_conversations(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir) / "project"
            root.mkdir()
            (root / "README.md").write_text("# Demo\n", encoding="utf-8")
            workspace = Workspace(root)
            store = ProjectMemoryStore(Path(temp_dir) / "memory")

            store.update_after_task(
                workspace,
                task="创建项目",
                answer="完成。\npython main.py",
                modified_files=["README.md"],
                latest_command="passed (exit 0)",
            )

            message = store.build_message(workspace)

            self.assertIsNotNone(message)
            content = str(message["content"])
            self.assertIn("<project_memory>", content)
            self.assertIn("README.md", content)
            self.assertIn("python main.py", content)
            self.assertIn("创建项目", content)

    def test_extracts_decisions_and_launch_commands(self) -> None:
        self.assertEqual(
            extract_user_decisions("User clarification:\n不要引入第三方依赖"),
            ["不要引入第三方依赖"],
        )
        self.assertEqual(
            extract_launch_commands("启动：\n```powershell\npython app.py\n```"),
            ["python app.py"],
        )


if __name__ == "__main__":
    unittest.main()
