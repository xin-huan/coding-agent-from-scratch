import unittest

from calculator.core import add, divide, multiply, subtract


class CalculatorTests(unittest.TestCase):
    def test_basic_operations(self) -> None:
        self.assertEqual(add(2, 3), 5)
        self.assertEqual(subtract(8, 3), 5)
        self.assertEqual(multiply(2, 4), 8)
        self.assertEqual(divide(10, 4), 2.5)

    def test_rejects_division_by_zero(self) -> None:
        with self.assertRaisesRegex(ValueError, "zero"):
            divide(1, 0)


if __name__ == "__main__":
    unittest.main()
