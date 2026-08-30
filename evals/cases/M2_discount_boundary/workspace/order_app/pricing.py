"""Order pricing rules."""

from __future__ import annotations


DISCOUNT_THRESHOLD = 100.0
DISCOUNT_RATE = 0.10


def discounted_total(prices: list[float]) -> float:
    if any(price < 0 for price in prices):
        raise ValueError("prices cannot be negative")
    subtotal = sum(prices)
    if subtotal > DISCOUNT_THRESHOLD:
        subtotal *= 1 - DISCOUNT_RATE
    return round(subtotal, 2)
