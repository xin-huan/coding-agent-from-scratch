import unittest

from todo_app.core import add_task, complete_task


class TodoTests(unittest.TestCase):
    def test_adds_and_completes_task(self) -> None:
        tasks = add_task([], "Write tests")
        self.assertEqual(complete_task(tasks, 1)[0]["done"], True)

    def test_rejects_missing_task(self) -> None:
        with self.assertRaisesRegex(ValueError, "does not exist"):
            complete_task([], 99)


if __name__ == "__main__":
    unittest.main()
