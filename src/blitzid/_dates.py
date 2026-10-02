"""Date coercion shared by the reading pipeline."""

from __future__ import annotations

import contextlib
from datetime import datetime

__all__ = ["parse_date"]

_DATE_FORMATS = ("%Y-%m-%d", "%d.%m.%Y")


def parse_date(value: object) -> object:
    """Parse date-shaped strings (ISO 8601 or DD.MM.YYYY) into dates."""
    if not isinstance(value, str):
        return value
    for fmt in _DATE_FORMATS:
        with contextlib.suppress(ValueError):
            return datetime.strptime(value, fmt).date()
    return value
