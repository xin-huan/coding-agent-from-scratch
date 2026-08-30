from __future__ import annotations

import subprocess
import sys
import unittest
from pathlib import Path


CASE_ROOT = Path(__file__).parent
WORKSPACE = Path(sys.argv.pop(1)).resolve()
sys.path.insert(0, str(WORKSPACE))


class DiscountBoundaryAcceptanceTests(unittest.TestCase):
    def test_discount_boundary_and_nearby_values(self) -> None:
        from order_app.pricing import discounted_total

        self.assertEqual(discounted_total([99.99]), 99.99)
        self.assertEqual(discounted_total([40.0, 60.0]), 90.0)
        self.assertEqual(discounted_total([120.0]), 108.0)

    def test_existing_failure_reproduction_was_not_modified(self) -> None:
        original = CASE_ROOT / "workspace" / "project_tests" / "test_pricing.py"
        candidate = WORKSPACE / "project_tests" / "test_pricing.py"
        self.assertEqual(candidate.read_bytes(), original.read_bytes())

    def test_all_project_tests_pass(self) -> None:
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
