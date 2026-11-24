"""Hosted plan catalog and dashboard-side limits.

The API enforces quotas and rate limits from ``PLAN_LIMITS`` in
``src/blitzid/api/_auth.py``; both limits are mirrored here for
the dashboard and a contract test in ``tests/test_api_auth.py`` fails
when the two drift. ``max_keys`` is dashboard-only.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Plan:
    """Published pricing and limits for a hosted plan.

    ``None`` means custom pricing or unlimited usage for Enterprise.
    """

    monthly_price_usd: int | None
    request_quota: int | None
    max_keys: int | None
    requests_per_minute: int | None


PLANS: dict[str, Plan] = {
    "starter": Plan(
        monthly_price_usd=5, request_quota=1_000, max_keys=1, requests_per_minute=30
    ),
    "pro": Plan(
        monthly_price_usd=19, request_quota=10_000, max_keys=5, requests_per_minute=120
    ),
    "enterprise": Plan(
        monthly_price_usd=None,
        request_quota=None,
        max_keys=None,
        requests_per_minute=None,
    ),
}
SELF_SERVE_PLANS = {
    name: plan for name, plan in PLANS.items() if plan.monthly_price_usd is not None
}


def limits_for(plan: str | None) -> Plan | None:
    """Return the limits for a plan name, or None when not subscribed."""
    return PLANS.get(plan or "")
