"""Pure entitlement rule mirrored by the API's cached-key check.

The web dashboard and the API enforce the same plan/status/cancellation
rule from separate packages. Keep this predicate dependency-free so the
API contract test can compare both implementations.
"""

from __future__ import annotations

from datetime import datetime
from typing import Final

ACTIVE_STATUSES: Final = frozenset({"active", "trialing"})


def entitled(
    status: str | None,
    cancel_at: datetime | None,
    grace_until: datetime | None,
    now: datetime,
) -> bool:
    """Return whether a subscription status grants access at ``now``."""
    if cancel_at is not None and cancel_at <= now:
        return False
    if status in ACTIVE_STATUSES:
        return True
    return status == "past_due" and grace_until is not None and grace_until > now
