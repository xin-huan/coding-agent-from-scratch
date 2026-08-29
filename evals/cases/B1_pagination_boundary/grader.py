from __future__ import annotations

import subprocess
import sys
import unittest
from pathlib import Path


WORKSPACE = Path(sys.argv.pop(1)).resolve()
sys.path.insert(0, str(WORKSPACE))


class PaginationAcceptanceTests(unittest.TestCase):
    def test_handles_first_last_exact_and_empty_pages(self) -> None:
        from pager import paginate

        self.assertEqual(paginate([1, 2, 3, 4, 5], 1, 2), [1, 2])
        self.assertEqual(paginate([1, 2, 3, 4, 5], 3, 2), [5])
        self.assertEqual(paginate([1, 2, 3, 4], 2, 2), [3, 4])
        self.assertEqual(paginate([], 1, 10), [])
        self.assertEqual(paginate([1], 2, 10), [])

    def test_rejects_invalid_boundaries(self) -> None:
        from pager import paginate

        for page, page_size in ((0, 1), (-1, 1), (1, 0), (1, -1)):
            with self.subTest(page=page, page_size=page_size):
                with self.assertRaises(ValueError):
                    paginate([1, 2], page, page_size)

    def test_project_tests_include_regression_and_pass(self) -> None:
        tests = "\n".join(
            path.read_text(encoding="utf-8")
            for path in (WORKSPACE / "project_tests").glob("test_*.py")
        )
        self.assertGreaterEqual(tests.count("def test_"), 2)
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
