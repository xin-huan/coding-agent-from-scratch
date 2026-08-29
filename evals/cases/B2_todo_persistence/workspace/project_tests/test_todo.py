import tempfile
import unittest
from pathlib import Path

from todo_app.core import add_task
from todo_app.storage import save_tasks


class TodoTests(unittest.TestCase):
    def test_adds_task(self) -> None:
        self.assertEqual(add_task([], "Write tests")[0]["title"], "Write tests")

    def test_save_creates_file(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "nested" / "tasks.json"
            save_tasks(path, [{"id": 1, "title": "Saved", "done": False}])
            self.assertTrue(path.exists())


if __name__ == "__main__":
    unittest.main()
