"""Date coercion shared by the reading pipeline.

Alongside the pipeline's ISO 8601, ``DD.MM.YYYY``, and ``DD/MM/YYYY``
shapes, the AAMVA
barcode reader feeds MMDDYYYY (U.S.) and CCYYMMDD (Canada) date
elements through these formats; ``%m%d%Y`` is tried first so a string
valid in both shapes (a US date) keeps the US reading.
"""

from __future__ import annotations

import contextlib
from datetime import datetime

__all__ = ["parse_date"]

_DATE_FORMATS = ("%Y-%m-%d", "%d.%m.%Y", "%d/%m/%Y", "%m%d%Y", "%Y%m%d")


def parse_date(value: object) -> object:
    """Parse an exact supported date string; return other values unchanged."""
    if not isinstance(value, str):
        return value
    for fmt in _DATE_FORMATS:
        with contextlib.suppress(ValueError):
            parsed = datetime.strptime(value, fmt).date()
            if parsed.strftime(fmt) == value:
                return parsed
    return value
