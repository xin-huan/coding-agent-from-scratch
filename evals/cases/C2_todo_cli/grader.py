from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


WORKSPACE = Path(sys.argv.pop(1)).resolve()


def run_todo(data_path: Path, *args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, "-m", "todo", "--data", str(data_path), *args],
        cwd=WORKSPACE,
        capture_output=True,
        text=True,
        timeout=5,
        check=False,
    )


class TodoAcceptanceTests(unittest.TestCase):
    def test_persists_add_list_and_done_across_processes(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            data_path = Path(temp_dir) / "tasks.json"

            added = run_todo(data_path, "add", "Write tests")
            listed = run_todo(data_path, "list")
            completed = run_todo(data_path, "done", "1")
            listed_again = run_todo(data_path, "list")

            for result in (added, listed, completed, listed_again):
                self.assertEqual(result.returncode, 0, result.stderr)
            self.assertIn("[ ] 1: Write tests", listed.stdout)
            self.assertIn("[x] 1: Write tests", listed_again.stdout)
            saved = json.loads(data_path.read_text(encoding="utf-8"))
            self.assertEqual(saved[0]["title"], "Write tests")
            self.assertTrue(saved[0]["done"])

    def test_rejects_invalid_input(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            data_path = Path(temp_dir) / "tasks.json"
            empty = run_todo(data_path, "add", "   ")
            missing = run_todo(data_path, "done", "99")
            self.assertNotEqual(empty.returncode, 0)
            self.assertNotEqual(missing.returncode, 0)

    def test_includes_documentation_and_passing_project_tests(self) -> None:
        self.assertTrue(any(WORKSPACE.glob("README*")))
        self.assertTrue(any((WORKSPACE / "tests").glob("test_*.py")))
        result = subprocess.run(
            [sys.executable, "-m", "unittest", "discover", "-s", "tests", "-v"],
            cwd=WORKSPACE,
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
        )
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)


if __name__ == "__main__":
    unittest.main()
