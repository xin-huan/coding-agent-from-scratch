import tempfile
import unittest
from pathlib import Path

from coding_agent.workspace import Workspace, WorkspaceError


class WorkspaceTests(unittest.TestCase):
    def test_resolves_relative_path_inside_workspace(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)

            resolved = Workspace(root).resolve("src/main.py")

            self.assertEqual(resolved, (root / "src" / "main.py").resolve())

    def test_rejects_path_outside_workspace(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            workspace = Workspace(Path(temp_dir))

            with self.assertRaisesRegex(WorkspaceError, "outside workspace"):
                workspace.resolve("../secret.txt")


if __name__ == "__main__":
    unittest.main()
