from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


WORKSPACE = Path(sys.argv.pop(1)).resolve()


def run_expense(data_path: Path, *args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, "-m", "expense_tracker", "--data", str(data_path), *args],
        cwd=WORKSPACE,
        capture_output=True,
        text=True,
        timeout=5,
        check=False,
    )


class ExpenseTrackerAcceptanceTests(unittest.TestCase):
    def test_add_list_total_and_json_persistence_across_processes(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            data_path = Path(temp_dir) / "expenses.json"
            first = run_expense(data_path, "add", "12.50", "Lunch")
            second = run_expense(data_path, "add", "7", "Bus")
            listed = run_expense(data_path, "list")
            total = run_expense(data_path, "total")

            for result in (first, second, listed, total):
                self.assertEqual(result.returncode, 0, result.stderr)
            self.assertIn("1 | 12.50 | Lunch", listed.stdout)
            self.assertIn("2 | 7.00 | Bus", listed.stdout)
            self.assertEqual(total.stdout.strip(), "19.50")
            saved = json.loads(data_path.read_text(encoding="utf-8"))
            self.assertEqual(len(saved), 2)
            self.assertEqual(saved[0]["description"], "Lunch")

    def test_rejects_invalid_amount_and_description(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            data_path = Path(temp_dir) / "expenses.json"
            for amount, description in (
                ("0", "Zero"),
                ("-1", "Negative"),
                ("abc", "Invalid"),
                ("1", "   "),
            ):
                with self.subTest(amount=amount, description=description):
                    result = run_expense(data_path, "add", amount, description)
                    self.assertNotEqual(result.returncode, 0)

    def test_has_required_modules_documentation_and_passing_tests(self) -> None:
        required = {
            "models.py",
            "storage.py",
            "service.py",
            "cli.py",
            "__main__.py",
        }
        package = WORKSPACE / "expense_tracker"
        self.assertTrue(required.issubset({path.name for path in package.glob("*.py")}))
        self.assertTrue(any(WORKSPACE.glob("README*")))
        tests = "\n".join(
            path.read_text(encoding="utf-8")
            for path in (WORKSPACE / "tests").glob("test_*.py")
        )
        self.assertGreaterEqual(tests.count("def test_"), 4)
        result = subprocess.run(
            [sys.executable, "-m", "unittest", "discover", "-s", "tests", "-v"],
            cwd=WORKSPACE,
            capture_output=True,
            text=True,
            timeout=15,
            check=False,
        )
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)


if __name__ == "__main__":
    unittest.main()
