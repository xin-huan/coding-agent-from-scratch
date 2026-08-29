import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


FIXTURE_PROJECT = Path(__file__).parent / "fixtures" / "todo_project"


class FixtureProjectTests(unittest.TestCase):
    def test_todo_fixture_starts_with_all_tests_passing(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            project = Path(temp_dir) / "todo_project"
            shutil.copytree(FIXTURE_PROJECT, project)

            result = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "unittest",
                    "discover",
                    "-s",
                    "project_tests",
                    "-v",
                ],
                cwd=project,
                capture_output=True,
                text=True,
                timeout=10,
                check=False,
            )

            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)


if __name__ == "__main__":
    unittest.main()
