"""Input validation rules."""


def validate_title(title: str) -> str:
    title = title.strip()
    if not title:
        raise ValueError("Title cannot be empty")
    if len(title) > 80:
        raise ValueError("Title is too long")
    return title
