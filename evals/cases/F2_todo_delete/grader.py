from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


WORKSPACE = Path(sys.argv.pop(1)).resolve()
sys.path.insert(0, str(WORKSPACE))


def run_todo(data_path: Path, *args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, "-m", "todo_app", "--data", str(data_path), *args],
        cwd=WORKSPACE,
        capture_output=True,
        text=True,
        timeout=5,
        check=False,
    )


class DeleteAcceptanceTests(unittest.TestCase):
    def test_core_deletes_only_the_selected_task(self) -> None:
        from todo_app.core import delete_task

        tasks = [
            {"id": 1, "title": "First", "done": False},
            {"id": 2, "title": "Second", "done": False},
        ]
        self.assertEqual(delete_task(tasks, 1), [tasks[1]])
        with self.assertRaises(ValueError):
            delete_task(tasks, 99)

    def test_cli_deletes_persisted_task_and_rejects_missing_id(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            data_path = Path(temp_dir) / "tasks.json"
            data_path.write_text(
                json.dumps([{"id": 1, "title": "Delete me", "done": False}]),
                encoding="utf-8",
            )
            deleted = run_todo(data_path, "delete", "1")
            missing = run_todo(data_path, "delete", "99")
            self.assertEqual(deleted.returncode, 0, deleted.stderr)
            self.assertEqual(json.loads(data_path.read_text(encoding="utf-8")), [])
            self.assertNotEqual(missing.returncode, 0)

    def test_original_and_new_project_tests_pass(self) -> None:
        test_text = "\n".join(
            path.read_text(encoding="utf-8")
            for path in (WORKSPACE / "project_tests").glob("test_*.py")
        )
        self.assertIn("delete", test_text)
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
