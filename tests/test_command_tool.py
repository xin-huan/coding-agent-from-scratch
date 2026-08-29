import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from coding_agent.tools.command import CommandToolError, run_command
from coding_agent.workspace import Workspace


class CommandToolTests(unittest.TestCase):
    def test_runs_python_script_inside_workspace(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            (root / "hello.py").write_text("print('hello')\n", encoding="utf-8")

            result = run_command(Workspace(root), ["python", "hello.py"])

            self.assertIn("Exit code: 0", result)
            self.assertIn("hello", result)

    def test_stops_command_after_timeout(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            (root / "slow.py").write_text(
                "import time\ntime.sleep(2)\n", encoding="utf-8"
            )

            with self.assertRaisesRegex(CommandToolError, "timed out"):
                run_command(
                    Workspace(root),
                    ["python", "slow.py"],
                    timeout_seconds=0.05,
                )

    def test_rejects_unapproved_executable(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            with self.assertRaisesRegex(CommandToolError, "not allowed"):
                run_command(Workspace(Path(temp_dir)), ["cmd", "/c", "dir"])

    def test_rejects_inline_python(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            with self.assertRaisesRegex(CommandToolError, "python -c"):
                run_command(
                    Workspace(Path(temp_dir)),
                    ["python", "-c", "print('unsafe')"],
                )

    def test_does_not_pass_api_key_to_child_process(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            (root / "show_key.py").write_text(
                "import os\nprint(os.environ.get('DEEPSEEK_API_KEY'))\n",
                encoding="utf-8",
            )

            with patch.dict(os.environ, {"DEEPSEEK_API_KEY": "test-secret"}):
                result = run_command(Workspace(root), ["python", "show_key.py"])

            self.assertNotIn("test-secret", result)
            self.assertIn("None", result)


if __name__ == "__main__":
    unittest.main()
