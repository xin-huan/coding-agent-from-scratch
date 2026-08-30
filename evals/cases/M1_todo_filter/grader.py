from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


WORKSPACE = Path(sys.argv.pop(1)).resolve()
sys.path.insert(0, str(WORKSPACE))


class TodoFilterAcceptanceTests(unittest.TestCase):
    def test_service_filters_case_insensitively_and_handles_empty_keyword(self) -> None:
        from todo_app.repository import TaskRepository
        from todo_app.service import TodoService

        with tempfile.TemporaryDirectory() as temp_dir:
            service = TodoService(TaskRepository(Path(temp_dir) / "tasks.json"))
            service.add("Read CODE")
            service.add("Buy milk")
            service.add("Review code")
            self.assertEqual(
                [task["title"] for task in service.list_tasks("code")],
                ["Read CODE", "Review code"],
            )
            self.assertEqual(len(service.list_tasks("   ")), 3)
            self.assertEqual(service.list_tasks("missing"), [])

    def test_cli_filters_without_changing_storage(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            data_path = Path(temp_dir) / "tasks.json"
            tasks = [
                {"id": 1, "title": "Read CODE", "done": False},
                {"id": 2, "title": "Buy milk", "done": False},
            ]
            data_path.write_text(json.dumps(tasks), encoding="utf-8")
            result = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "todo_app",
                    "--data",
                    str(data_path),
                    "list",
                    "--keyword",
                    "code",
                ],
                cwd=WORKSPACE,
                capture_output=True,
                text=True,
                timeout=5,
                check=False,
            )
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertIn("Read CODE", result.stdout)
            self.assertNotIn("Buy milk", result.stdout)
            self.assertEqual(json.loads(data_path.read_text(encoding="utf-8")), tasks)

    def test_original_and_new_project_tests_pass(self) -> None:
        tests = "\n".join(
            path.read_text(encoding="utf-8")
            for path in (WORKSPACE / "project_tests").glob("test_*.py")
        )
        self.assertGreaterEqual(tests.count("def test_"), 2)
        result = subprocess.run(
            [sys.executable, "-m", "unittest", "discover", "-s", "project_tests", "-v"],
            cwd=WORKSPACE,
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
        )
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)


if __name__ == "__main__":
    unittest.main()
