import tempfile
import unittest
from pathlib import Path

from todo_app.core import add_task, complete_task
from todo_app.storage import load_tasks, save_tasks


class TodoTests(unittest.TestCase):
    def test_adds_and_completes_a_task(self) -> None:
        tasks = add_task([], "Write tests")

        completed = complete_task(tasks, 1)

        self.assertEqual(
            completed,
            [{"id": 1, "title": "Write tests", "done": True}],
        )

    def test_rejects_an_empty_title(self) -> None:
        with self.assertRaisesRegex(ValueError, "cannot be empty"):
            add_task([], "   ")

    def test_saves_and_loads_tasks(self) -> None:
        tasks = [{"id": 1, "title": "Persist me", "done": False}]
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "tasks.json"

            save_tasks(path, tasks)

            self.assertEqual(load_tasks(path), tasks)


if __name__ == "__main__":
    unittest.main()
