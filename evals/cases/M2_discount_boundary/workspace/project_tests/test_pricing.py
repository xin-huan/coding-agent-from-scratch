import unittest

from order_app.pricing import discounted_total


class PricingTests(unittest.TestCase):
    def test_below_threshold_has_no_discount(self) -> None:
        self.assertEqual(discounted_total([40.0, 59.99]), 99.99)

    def test_threshold_order_receives_discount(self) -> None:
        self.assertEqual(discounted_total([40.0, 60.0]), 90.0)

    def test_rejects_negative_prices(self) -> None:
        with self.assertRaises(ValueError):
            discounted_total([10.0, -1.0])


if __name__ == "__main__":
    unittest.main()
