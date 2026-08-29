"""Calculator command-line interface."""

from __future__ import annotations

import argparse
from typing import Sequence

from calculator.core import OPERATIONS, calculate


def _format_number(value: float) -> str:
    return str(int(value)) if value.is_integer() else str(value)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Mini Calculator")
    parser.add_argument("operation", choices=OPERATIONS)
    parser.add_argument("left", type=float)
    parser.add_argument("right", type=float)
    args = parser.parse_args(argv)
    try:
        result = calculate(args.operation, args.left, args.right)
    except ValueError as error:
        parser.error(str(error))
    print(_format_number(result))
    return 0
