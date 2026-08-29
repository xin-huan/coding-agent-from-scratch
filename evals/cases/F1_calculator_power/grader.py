from __future__ import annotations

import subprocess
import sys
import unittest
from pathlib import Path


WORKSPACE = Path(sys.argv.pop(1)).resolve()
sys.path.insert(0, str(WORKSPACE))


class PowerAcceptanceTests(unittest.TestCase):
    def test_core_supports_power(self) -> None:
        from calculator.core import power

        self.assertEqual(power(2, 3), 8)
        self.assertAlmostEqual(power(4, -0.5), 0.5)
        self.assertAlmostEqual(power(2.5, 2), 6.25)

    def test_cli_supports_power(self) -> None:
        result = subprocess.run(
            [sys.executable, "-m", "calculator", "power", "2", "3"],
            cwd=WORKSPACE,
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(result.stdout.strip(), "8")

    def test_original_and_new_project_tests_pass(self) -> None:
        test_text = "\n".join(
            path.read_text(encoding="utf-8")
            for path in (WORKSPACE / "project_tests").glob("test_*.py")
        )
        self.assertIn("power", test_text)
        result = subprocess.run(
            [sys.executable, "-m", "unittest", "discover", "-s", "project_tests", "-v"],
            cwd=WORKSPACE,
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
        )
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)


if __name__ == "__main__":
    unittest.main()
