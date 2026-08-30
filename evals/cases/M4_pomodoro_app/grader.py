from __future__ import annotations

import re
import subprocess
import sys
import unittest
from pathlib import Path


WORKSPACE = Path(sys.argv.pop(1)).resolve()


def project_sources() -> list[Path]:
    return [
        path
        for path in WORKSPACE.rglob("*.py")
        if "tests" not in path.parts
        and "__pycache__" not in path.parts
        and not path.name.startswith("test_")
        and not path.name.startswith("_smoke")
    ]


def project_tests() -> list[Path]:
    return [
        path
        for path in WORKSPACE.rglob("test_*.py")
        if "__pycache__" not in path.parts
    ]


def combined(paths: list[Path]) -> str:
    return "\n".join(path.read_text(encoding="utf-8") for path in paths).lower()


class PomodoroAcceptanceTests(unittest.TestCase):
    def test_is_a_runnable_multi_file_desktop_application(self) -> None:
        sources = project_sources()
        self.assertGreaterEqual(len(sources), 3)
        self.assertTrue(
            any(
                path.name in {"__main__.py", "main.py"}
                or (
                    '__name__' in path.read_text(encoding="utf-8")
                    and '__main__' in path.read_text(encoding="utf-8")
                )
                for path in sources
            ),
            "A Python launch entry point is required",
        )
        source = combined(sources)
        self.assertTrue(
            any(name in source for name in ("tkinter", "pyside", "pyqt", "wx")),
            "No supported desktop GUI framework was found",
        )
        compiled = subprocess.run(
            [sys.executable, "-m", "compileall", "-q", "."],
            cwd=WORKSPACE,
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
        )
        self.assertEqual(compiled.returncode, 0, compiled.stdout + compiled.stderr)

    def test_exposes_required_controls_and_non_blocking_timer_updates(self) -> None:
        source = combined(project_sources())
        terms = {
            "start": ("start", "\u5f00\u59cb"),
            "pause": ("pause", "\u6682\u505c"),
            "reset": ("reset", "\u91cd\u7f6e"),
            "skip": ("skip", "\u8df3\u8fc7"),
            "work": ("work", "\u5de5\u4f5c"),
            "break": ("break", "rest", "\u4f11\u606f"),
            "remaining": ("remaining", "time_left", "\u5269\u4f59"),
        }
        for capability, alternatives in terms.items():
            with self.subTest(capability=capability):
                self.assertTrue(
                    any(term in source for term in alternatives),
                    f"Missing capability: {capability}",
                )
        self.assertTrue(
            any(widget in source for widget in ("spinbox", "entry", "scale")),
            "No editable duration control was found",
        )
        self.assertTrue(".after(" in source or "qtimer" in source)
        self.assertNotIn("time.sleep(", source)

    def test_separates_testable_timer_logic_from_the_gui(self) -> None:
        sources = project_sources()
        logic_files = [
            path
            for path in sources
            if path.name not in {"__init__.py", "__main__.py", "main.py"}
            and not any(gui in path.read_text(encoding="utf-8").lower()
                        for gui in ("tkinter", "pyside", "pyqt", "wx"))
            and any(term in path.read_text(encoding="utf-8").lower()
                    for term in ("remaining", "time_left", "tick", "phase"))
        ]
        self.assertTrue(logic_files, "Timer logic is not separated from GUI code")
        test_source = combined(project_tests())
        self.assertTrue(
            any(path.stem.lower() in test_source for path in logic_files),
            "Tests do not exercise the separated timer logic",
        )

    def test_provides_passing_automated_tests_and_launch_instructions(self) -> None:
        readmes = list(WORKSPACE.glob("README*"))
        self.assertTrue(readmes, "README is missing")
        readme = combined(readmes)
        self.assertIn("python", readme)
        self.assertTrue("-m" in readme or "main.py" in readme)

        tests = project_tests()
        test_source = combined(tests)
        self.assertGreaterEqual(len(re.findall(r"def\s+test_", test_source)), 4)
        result = subprocess.run(
            [sys.executable, "-m", "unittest", "discover", "-v"],
            cwd=WORKSPACE,
            capture_output=True,
            text=True,
            timeout=15,
            check=False,
        )
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)


if __name__ == "__main__":
    unittest.main()
