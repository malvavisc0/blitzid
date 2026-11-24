"""Hosted API-key enforcement tests (fake resolver, no Postgres)."""

from __future__ import annotations

import importlib.util
import sys
from dataclasses import replace
from datetime import UTC, datetime, timedelta
from pathlib import Path
from types import ModuleType
from typing import Any

import fakeredis
import pytest
from fastapi.testclient import TestClient
from test_api import MemoryJobStore, _stub_engines

from blitzid.api import _auth
from blitzid.api._app import create_app
from blitzid.api._auth import (
    PLAN_LIMITS,
    SaasError,
    _entitled,
    build_api_auth,
    parse_authorization,
)
from blitzid.exceptions import BlitzIDError

VALID_KEY = "Bearer blz_live_abcdefgh." + "s" * 43


class FakeAuth:
    def __init__(self, error: SaasError | None = None, ready: bool = True) -> None:
        self.error = error
        self.ready_value = ready

    def authenticate_and_enforce(self, authorization: str | None) -> None:
        if self.error is not None:
            raise self.error
        if authorization != VALID_KEY:
            raise SaasError(401, "invalid_api_key", "Invalid API key.")

    def ready(self) -> bool:
        return self.ready_value


@pytest.fixture
def auth_client() -> Any:
    app = create_app(store=MemoryJobStore(), engines=_stub_engines())
    with TestClient(app) as client:
        app.state.config = replace(app.state.config, auth_required=True)
        app.state.api_auth = FakeAuth()
        yield client


def test_health_is_public_when_auth_is_enabled(auth_client: TestClient) -> None:
    response = auth_client.get("/api/health")
    assert response.status_code == 200
    assert response.json()["auth"] is True


def test_protected_route_requires_a_key(auth_client: TestClient) -> None:
    response = auth_client.post("/api/analyze")
    assert response.status_code == 401
    assert response.json() == {
        "error": {
            "code": "invalid_api_key",
            "message": "Invalid API key.",
        }
    }


def test_jobs_route_requires_a_key(auth_client: TestClient) -> None:
    assert auth_client.get("/api/jobs/abc").status_code == 401


def test_valid_key_reaches_normal_validation(auth_client: TestClient) -> None:
    response = auth_client.post("/api/analyze", headers={"Authorization": VALID_KEY})
    assert response.status_code == 422


def test_plan_and_rate_errors_are_structured(auth_client: TestClient) -> None:
    auth_client.app.state.api_auth = FakeAuth(
        SaasError(403, "plan_required", "An active plan is required.")
    )
    response = auth_client.post("/api/analyze", headers={"Authorization": VALID_KEY})
    assert response.status_code == 403
    assert response.json()["error"]["code"] == "plan_required"

    auth_client.app.state.api_auth = FakeAuth(
        SaasError(
            429, "quota_exceeded", "Monthly request limit reached.", retry_after=42
        )
    )
    response = auth_client.post("/api/analyze", headers={"Authorization": VALID_KEY})
    assert response.status_code == 429
    assert response.headers["Retry-After"] == "42"
    assert response.json()["error"]["code"] == "quota_exceeded"


def test_auth_unavailable_fails_closed(auth_client: TestClient) -> None:
    auth_client.app.state.api_auth = FakeAuth(
        SaasError(
            503, "auth_unavailable", "Authentication is temporarily unavailable."
        ),
        ready=False,
    )
    response = auth_client.post("/api/analyze", headers={"Authorization": VALID_KEY})
    assert response.status_code == 503
    assert response.json()["error"]["code"] == "auth_unavailable"
    health = auth_client.get("/api/health")
    assert health.status_code == 503
    assert health.json()["auth"] is False


class UsageFakeAuth(FakeAuth):
    def __init__(self) -> None:
        super().__init__()
        self.recorded: list[tuple[dict[str, Any], datetime]] = []

    def authenticate_and_enforce(
        self, authorization: str | None
    ) -> tuple[dict[str, Any], datetime]:
        return {"key_id": "key_id"}, datetime(2027, 1, 1, tzinfo=UTC)

    def record_usage(self, record: dict[str, Any], now: datetime) -> None:
        self.recorded.append((record, now))


def test_usage_is_recorded_after_the_response(auth_client: TestClient) -> None:
    fake = UsageFakeAuth()
    auth_client.app.state.api_auth = fake
    response = auth_client.post(
        "/api/liveness/challenge", headers={"Authorization": VALID_KEY}
    )
    assert response.status_code == 200
    assert fake.recorded == [({"key_id": "key_id"}, datetime(2027, 1, 1, tzinfo=UTC))]


def test_authorization_parsing() -> None:
    prefix, secret = parse_authorization(VALID_KEY)
    assert prefix == "abcdefgh"
    assert secret == "s" * 43
    for bad in (
        None,
        "",
        "Bearer",
        "Bearer nope",
        "Bearer blz_live_short.s",
        "Bearer blz_live_abcdefgh.missingdot",
    ):
        with pytest.raises(SaasError) as exc:
            parse_authorization(bad)
        assert exc.value.code == "invalid_api_key"


def test_self_hosted_default_remains_open() -> None:
    app = create_app(store=MemoryJobStore(), engines=_stub_engines())
    with TestClient(app) as client:
        response = client.post("/api/analyze")
        assert response.status_code == 422


def test_build_api_auth_requires_a_database_url() -> None:
    with pytest.raises(BlitzIDError, match="BLITZID_SAAS_DATABASE_URL"):
        build_api_auth("", "redis://localhost:6379/0")


def test_unknown_prefix_is_negatively_cached(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    resolver = object.__new__(_auth.PostgresApiAuth)
    resolver._redis = fakeredis.FakeRedis(decode_responses=True)
    calls: list[str] = []

    def missing(prefix: str) -> None:
        calls.append(prefix)

    monkeypatch.setattr(resolver, "_lookup", missing)
    key = "Bearer blz_live_unknown1." + "s" * 43
    for _ in range(2):
        with pytest.raises(SaasError) as exc:
            resolver.authenticate_and_enforce(key)
        assert exc.value.code == "invalid_api_key"
    assert calls == ["unknown1"]


def test_lookup_flood_is_rate_limited(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    resolver = object.__new__(_auth.PostgresApiAuth)
    resolver._redis = fakeredis.FakeRedis(decode_responses=True)
    monkeypatch.setattr(_auth, "_LOOKUP_LIMIT_PER_MINUTE", 1)
    now = datetime(2027, 1, 1, 0, 0, 30, tzinfo=UTC)
    resolver._allow_lookup(now)
    with pytest.raises(SaasError) as exc:
        resolver._allow_lookup(now)
    assert exc.value.code == "rate_limit_exceeded"
    assert exc.value.status_code == 429


def test_cached_entitlement_ends_at_scheduled_cancellation() -> None:
    end = datetime(2027, 1, 1, tzinfo=UTC)
    record = {
        "plan": "starter",
        "subscription_status": "active",
        "cancel_at": end.isoformat(),
    }
    assert _entitled(record, end - timedelta(seconds=1))
    assert not _entitled(record, end)


def _load_web_plans() -> ModuleType:
    path = Path(__file__).resolve().parents[1] / "web" / "app" / "plans.py"
    spec = importlib.util.spec_from_file_location("web_plans", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_web_plan_limits_match_api_enforcement() -> None:
    """The dashboard must show the same plans and limits the API enforces."""
    web_plans = _load_web_plans()
    assert set(web_plans.PLANS) == set(PLAN_LIMITS)
    for plan, (quota, per_minute) in PLAN_LIMITS.items():
        assert web_plans.PLANS[plan].request_quota == quota
        assert web_plans.PLANS[plan].requests_per_minute == per_minute


def _load_web_entitlement() -> ModuleType:
    path = Path(__file__).resolve().parents[1] / "web" / "app" / "entitlement.py"
    spec = importlib.util.spec_from_file_location("web_entitlement", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _load_web_cache() -> ModuleType:
    path = Path(__file__).resolve().parents[1] / "web" / "app" / "cache.py"
    spec = importlib.util.spec_from_file_location("web_cache", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_web_cache_invalidates_api_key_cache_entries() -> None:
    """The web invalidator and the API resolver must share key namespaces."""
    redis_client = fakeredis.FakeRedis(decode_responses=True)
    resolver = object.__new__(_auth.PostgresApiAuth)
    resolver._redis = redis_client
    org_id = "00000000-0000-0000-0000-000000000001"
    resolver._cache_set("abcdefgh", {"org_id": org_id, "key_id": "key_id"})
    assert redis_client.exists("apikey:abcdefgh")

    cache = _load_web_cache().KeyCache(redis_client)
    cache.invalidate_org(org_id)

    assert not redis_client.exists("apikey:abcdefgh")
    assert not redis_client.exists(f"orgkeys:{org_id}")


def test_web_entitlement_matches_api_enforcement() -> None:
    """The dashboard and API must agree on status, grace, and cancellation."""
    web = _load_web_entitlement()
    now = datetime(2027, 1, 1, tzinfo=UTC)
    for status in (
        "active",
        "trialing",
        "past_due",
        "canceled",
        "unpaid",
        "incomplete",
    ):
        for cancel_at in (None, now - timedelta(days=1), now + timedelta(days=1)):
            for grace in (None, now - timedelta(days=1), now + timedelta(days=1)):
                record = {
                    "plan": "starter",
                    "subscription_status": status,
                    "cancel_at": cancel_at.isoformat() if cancel_at else None,
                    "grace_until": grace.isoformat() if grace else None,
                }
                assert _entitled(record, now) == web.entitled(
                    status, cancel_at, grace, now
                ), (status, cancel_at, grace)
