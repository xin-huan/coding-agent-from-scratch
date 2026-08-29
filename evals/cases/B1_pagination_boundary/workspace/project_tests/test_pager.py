import unittest

from pager import paginate


class PagerTests(unittest.TestCase):
    def test_returns_full_pages(self) -> None:
        items = ["a", "b", "c", "d", "e"]
        self.assertEqual(paginate(items, 1, 2), ["a", "b"])
        self.assertEqual(paginate(items, 2, 2), ["c", "d"])


if __name__ == "__main__":
    unittest.main()
