"""Shared token-ID validation helpers used by runtime and profiling contracts."""

from __future__ import annotations


def validate_token_ids(name: str, values: object, *, allow_empty: bool = False) -> None:
    """Validate a tuple of non-negative integers, raising ValueError on failure.

    When *allow_empty* is False (default), the tuple must be non-empty.
    """
    if not isinstance(values, tuple) or (not allow_empty and not values):
        qualifier = "a tuple" if allow_empty else "a non-empty tuple"
        raise ValueError(f"{name} must be {qualifier}")
    if any(type(token_id) is not int or token_id < 0 for token_id in values):
        raise ValueError(f"{name} must contain only non-negative integer tokens")