import tempfile
import unittest
from pathlib import Path

from notes_app.repository import NoteRepository
from notes_app.service import NoteService


class NoteTests(unittest.TestCase):
    def test_validates_and_saves_note(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            service = NoteService(NoteRepository(Path(temp_dir) / "notes.json"))
            service.add("Read code")
            self.assertEqual(service.repository.load()[0]["title"], "Read code")

    def test_rejects_empty_title(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            service = NoteService(NoteRepository(Path(temp_dir) / "notes.json"))
            with self.assertRaises(ValueError):
                service.add("   ")


if __name__ == "__main__":
    unittest.main()
