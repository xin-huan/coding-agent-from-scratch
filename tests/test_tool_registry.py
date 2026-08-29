import tempfile
import unittest
from pathlib import Path

from coding_agent.tools.registry import ToolRegistry
from coding_agent.workspace import Workspace


class ToolRegistryTests(unittest.TestCase):
    def test_exposes_exactly_the_six_mvp_tools(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            definitions = ToolRegistry(Workspace(Path(temp_dir))).definitions

        names: set[object] = set()
        for definition in definitions:
            function = definition["function"]
            self.assertIsInstance(function, dict)
            names.add(function["name"])
        self.assertEqual(
            names,
            {
                "list_files",
                "read_file",
                "search_text",
                "write_file",
                "apply_patch",
                "run_command",
            },
        )


if __name__ == "__main__":
    unittest.main()
