"""Date coercion shared by the reading pipeline.

Alongside the pipeline's ISO 8601 and ``DD.MM.YYYY`` shapes, the AAMVA
barcode reader feeds MMDDYYYY (U.S.) and CCYYMMDD (Canada) date
elements through these formats; ``%m%d%Y`` is tried first so a string
valid in both shapes (a US date) keeps the US reading.
"""

from __future__ import annotations

import contextlib
from datetime import datetime

__all__ = ["parse_date"]

_DATE_FORMATS = ("%Y-%m-%d", "%d.%m.%Y", "%m%d%Y", "%Y%m%d")


def parse_date(value: object) -> object:
    """Parse date-shaped strings (ISO 8601 or DD.MM.YYYY) into dates."""
    if not isinstance(value, str):
        return value
    for fmt in _DATE_FORMATS:
        with contextlib.suppress(ValueError):
            return datetime.strptime(value, fmt).date()
    return value
