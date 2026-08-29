from __future__ import annotations

import subprocess
import sys
import unittest
from pathlib import Path


WORKSPACE = Path(sys.argv.pop(1)).resolve()


def run_calculator(*args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, "-m", "calculator", *args],
        cwd=WORKSPACE,
        capture_output=True,
        text=True,
        timeout=5,
        check=False,
    )


class CalculatorAcceptanceTests(unittest.TestCase):
    def test_supports_required_operations(self) -> None:
        examples = {
            ("add", "2", "3"): "5",
            ("subtract", "8", "3"): "5",
            ("multiply", "2.5", "4"): "10",
            ("divide", "10", "4"): "2.5",
        }
        for arguments, expected in examples.items():
            with self.subTest(operation=arguments[0]):
                result = run_calculator(*arguments)
                self.assertEqual(result.returncode, 0, result.stderr)
                self.assertEqual(result.stdout.strip(), expected)

    def test_rejects_invalid_input(self) -> None:
        for arguments in (
            ("divide", "1", "0"),
            ("unknown", "1", "2"),
            ("add", "one", "2"),
        ):
            with self.subTest(arguments=arguments):
                result = run_calculator(*arguments)
                self.assertNotEqual(result.returncode, 0)
                self.assertTrue((result.stdout + result.stderr).strip())

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
