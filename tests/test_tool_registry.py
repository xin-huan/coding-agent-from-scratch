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

    def test_command_description_names_allowed_programs_and_restrictions(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            definitions = ToolRegistry(Workspace(Path(temp_dir))).definitions

        command = next(
            definition["function"]
            for definition in definitions
            if definition["function"]["name"] == "run_command"
        )
        description = command["description"]
        self.assertIn("python", description)
        self.assertIn("pytest", description)
        self.assertIn("git", description)
        self.assertIn("Do not use", description)
        self.assertIn("python -c", description)

    def test_reuses_unchanged_read_and_invalidates_after_write(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            (root / "module.py").write_text("VALUE = 1\n", encoding="utf-8")
            registry = ToolRegistry(Workspace(root))

            first = registry.execute("read_file", {"path": "module.py"})
            second = registry.execute("read_file", {"path": "module.py"})
            registry.execute(
                "write_file",
                {"path": "module.py", "content": "VALUE = 2\n"},
            )
            third = registry.execute("read_file", {"path": "module.py"})

        self.assertIn("VALUE = 1", first)
        self.assertIn("unchanged read cache hit", second)
        self.assertNotIn("VALUE = 1", second)
        self.assertIn("VALUE = 2", third)
        self.assertEqual(registry.read_cache_hits, 1)


if __name__ == "__main__":
    unittest.main()
