from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


WORKSPACE = Path(sys.argv.pop(1)).resolve()
sys.path.insert(0, str(WORKSPACE))


class PersistenceAcceptanceTests(unittest.TestCase):
    def test_save_writes_json_that_loads_in_a_new_call(self) -> None:
        from todo_app.storage import load_tasks, save_tasks

        tasks = [{"id": 1, "title": "Persist me", "done": False}]
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "nested" / "tasks.json"
            save_tasks(path, tasks)
            self.assertEqual(json.loads(path.read_text(encoding="utf-8")), tasks)
            self.assertEqual(load_tasks(path), tasks)

    def test_cli_can_read_data_written_by_an_earlier_process(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "tasks.json"
            add = subprocess.run(
                [sys.executable, "-m", "todo_app", "--data", str(path), "add", "Restart me"],
                cwd=WORKSPACE,
                capture_output=True,
                text=True,
                timeout=5,
                check=False,
            )
            listed = subprocess.run(
                [sys.executable, "-m", "todo_app", "--data", str(path), "list"],
                cwd=WORKSPACE,
                capture_output=True,
                text=True,
                timeout=5,
                check=False,
            )
            self.assertEqual(add.returncode, 0, add.stderr)
            self.assertEqual(listed.returncode, 0, listed.stderr)
            self.assertIn("Restart me", listed.stdout)

    def test_project_tests_include_regression_and_pass(self) -> None:
        tests = "\n".join(
            path.read_text(encoding="utf-8")
            for path in (WORKSPACE / "project_tests").glob("test_*.py")
        )
        self.assertGreaterEqual(tests.count("def test_"), 3)
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
