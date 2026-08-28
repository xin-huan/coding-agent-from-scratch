import io
import os
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from unittest.mock import patch

from coding_agent.cli import main


class CliTests(unittest.TestCase):
    def test_user_can_start_and_exit_a_workspace_session(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            output = io.StringIO()
            environment = {
                "DEEPSEEK_API_KEY": "test-secret",
                "DEEPSEEK_MODEL": "test-model",
            }

            with patch.dict(os.environ, environment, clear=True):
                with patch("coding_agent.cli.Path.cwd", return_value=Path(temp_dir)):
                    with patch("builtins.input", return_value="/exit"):
                        with redirect_stdout(output):
                            exit_code = main(["--workspace", temp_dir])

            self.assertEqual(exit_code, 0)
            self.assertIn(str(Path(temp_dir).resolve()), output.getvalue())
            self.assertIn("test-model", output.getvalue())


if __name__ == "__main__":
    unittest.main()
