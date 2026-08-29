"""Application entry and command dispatch."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Sequence

from notes_app.config import Settings
from notes_app.repository import NoteRepository
from notes_app.service import NoteService


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Configured Notes")
    parser.add_argument("--config", type=Path, default=Path("config.json"))
    parser.add_argument("command", choices=("add",))
    parser.add_argument("title")
    args = parser.parse_args(argv)

    settings = Settings.load(args.config)
    service = NoteService(NoteRepository(settings.data_path))
    try:
        service.add(args.title)
    except ValueError as error:
        parser.error(str(error))
    return 0
