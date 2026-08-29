"""Pagination helper."""

from __future__ import annotations

from typing import TypeVar


Item = TypeVar("Item")


def paginate(items: list[Item], page: int, page_size: int) -> list[Item]:
    if page < 1:
        raise ValueError("page must be at least 1")
    if page_size < 1:
        raise ValueError("page_size must be at least 1")

    start = (page - 1) * page_size
    end = min(start + page_size, len(items) - 1)
    return items[start:end]
