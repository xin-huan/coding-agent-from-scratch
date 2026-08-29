"""Note use cases and error propagation."""

from __future__ import annotations

from notes_app.repository import NoteRepository
from notes_app.validation import validate_title


class NoteService:
    def __init__(self, repository: NoteRepository) -> None:
        self.repository = repository

    def add(self, title: str) -> dict[str, object]:
        notes = self.repository.load()
        note = {"id": len(notes) + 1, "title": validate_title(title)}
        notes.append(note)
        self.repository.save(notes)
        return note
