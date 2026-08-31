import tempfile
import unittest
from pathlib import Path

from coding_agent.skills import SkillRegistry
from coding_agent.skills.registry import load_skills


class SkillRegistryTests(unittest.TestCase):
    def test_loads_builtin_skills(self) -> None:
        registry = SkillRegistry.load_builtin()
        names = {skill.name for skill in registry.skills}

        self.assertIn("diagnose", names)
        self.assertIn("verification", names)
        self.assertIn("desktop-python", names)
        self.assertIn("web-ui", names)
        self.assertIn("project-summary", names)

    def test_selects_relevant_skills_by_trigger_score(self) -> None:
        registry = SkillRegistry.load_builtin()

        selected = registry.select(
            "Tkinter 番茄钟运行时报错 Traceback NameError，请修复",
            limit=2,
        )

        self.assertEqual([skill.name for skill in selected], ["diagnose", "desktop-python"])
        self.assertIn("<skill name=\"diagnose\">", selected[0].message()["content"])

    def test_loads_markdown_skills_from_directory(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "custom.md"
            path.write_text(
                "---\n"
                "name: custom\n"
                "description: Custom workflow\n"
                "triggers: alpha, beta\n"
                "---\n"
                "Follow the custom workflow.\n",
                encoding="utf-8",
            )

            skills = load_skills(Path(temp_dir))

            self.assertEqual(len(skills), 1)
            self.assertEqual(skills[0].name, "custom")
            self.assertEqual(skills[0].triggers, ("alpha", "beta"))
            self.assertIn("custom workflow", skills[0].instructions)


if __name__ == "__main__":
    unittest.main()
