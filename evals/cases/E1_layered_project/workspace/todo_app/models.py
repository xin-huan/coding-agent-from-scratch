"""Shared task data shape."""

from typing import TypedDict


class Task(TypedDict):
    id: int
    title: str
    done: bool
