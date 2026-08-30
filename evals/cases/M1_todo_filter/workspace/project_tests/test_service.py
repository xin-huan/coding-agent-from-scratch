import tempfile
import unittest
from pathlib import Path

from todo_app.repository import TaskRepository
from todo_app.service import TodoService


class ServiceTests(unittest.TestCase):
    def test_adds_and_lists_task(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            service = TodoService(TaskRepository(Path(temp_dir) / "tasks.json"))
            service.add("Read code")
            self.assertEqual(service.list_tasks()[0]["title"], "Read code")


if __name__ == "__main__":
    unittest.main()
