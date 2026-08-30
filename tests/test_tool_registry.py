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

    def test_path_parameters_explain_workspace_relative_format(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            definitions = ToolRegistry(Workspace(Path(temp_dir))).definitions

        for definition in definitions:
            function = definition["function"]
            properties = function["parameters"]["properties"]
            for name in ("path", "cwd"):
                if name not in properties:
                    continue
                description = properties[name].get("description", "")
                self.assertIn("For example", description)
                self.assertIn("Never use", description)


if __name__ == "__main__":
    unittest.main()
