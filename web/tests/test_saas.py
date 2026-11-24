"""Tests for OAuth sessions, API keys, billing, and the dashboard."""

from __future__ import annotations

import json
import re
from dataclasses import replace
from datetime import UTC, datetime, timedelta
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest.mock import Mock

import fakeredis
import pytest
import stripe
from fastapi.responses import RedirectResponse
from fastapi.testclient import TestClient
from sqlalchemy import func, select

from app import auth
from app.billing import (
    BillingError,
    BillingEvent,
    NullBilling,
    StripeBilling,
    SubscriptionSnapshot,
)
from app.cache import KeyCache
from app.config import Settings
from app.main import create_app
from app.models import (
    ApiKey,
    Identity,
    Membership,
    Organization,
    StripeEvent,
    UsageDaily,
    User,
    utcnow,
)

CSRF_RE = re.compile(r'name="csrf_token" value="([^"]+)"')
KEY_RE = re.compile(r"blz_live_[A-Za-z0-9_-]{8}\.[A-Za-z0-9_-]+")


class FakeProvider:
    name = "google"

    def __init__(self, profile: auth.OAuthProfile | None = None) -> None:
        self.profile = profile or auth.OAuthProfile(
            provider="google",
            subject="sub-123",
            email="dev@example.com",
            email_verified=True,
            name="Dev Example",
            avatar_url=None,
        )

    async def authorize_redirect(self, request: Any) -> RedirectResponse:
        return RedirectResponse("/fake-oauth", status_code=303)

    async def fetch_profile(self, request: Any) -> auth.OAuthProfile:
        return self.profile


class BrokenProvider(FakeProvider):
    async def fetch_profile(self, request: Any) -> auth.OAuthProfile:
        raise auth.AuthError("provider failed")


class FakeBilling:
    def __init__(self) -> None:
        self.customer_id = "cus_test"
        self.subscription = SubscriptionSnapshot(
            status="active",
            price_id="price_starter",
            current_period_end=datetime(2027, 1, 1, tzinfo=UTC),
            customer_id="cus_test",
        )
        self.checkout_args: dict[str, Any] | None = None
        self.plan_change_args: tuple[str, str, str, str] | None = None
        self.cancel_args: tuple[str, str] | None = None

    def ensure_customer(self, org_id: str, email: str, name: str | None) -> str:
        return self.customer_id

    def create_checkout(
        self,
        customer_id: str,
        plan: str,
        org_id: str,
        success_url: str,
        cancel_url: str,
    ) -> str:
        self.checkout_args = {
            "customer": customer_id,
            "plan": plan,
            "org_id": org_id,
            "success": success_url,
            "cancel": cancel_url,
        }
        return "https://checkout.test/session"

    def create_portal(self, customer_id: str, return_url: str) -> str:
        return "https://portal.test/session"

    def create_plan_change(
        self, customer_id: str, current_plan: str, plan: str, return_url: str
    ) -> str:
        self.plan_change_args = customer_id, current_plan, plan, return_url
        return "https://portal.test/confirm-plan"

    def create_cancellation(self, customer_id: str, return_url: str) -> str:
        self.cancel_args = customer_id, return_url
        return "https://portal.test/confirm-cancellation"

    def parse_event(self, payload: bytes, signature: str) -> BillingEvent:
        if signature != "test-signature":
            raise ValueError("bad signature")
        data = json.loads(payload)
        return BillingEvent(
            id=data["id"], type=data["type"], object=data["data"]["object"]
        )

    def fetch_subscription(self, subscription_id: str) -> SubscriptionSnapshot:
        return self.subscription


@pytest.fixture
def schema_file(tmp_path: Path) -> Path:
    path = tmp_path / "openapi.json"
    path.write_text(
        json.dumps(
            {
                "openapi": "3.1.0",
                "info": {"title": "BlitzID API", "version": "9.9.9"},
                "paths": {},
            }
        ),
        encoding="utf-8",
    )
    return path


@pytest.fixture
def app(schema_file: Path) -> Any:
    settings = Settings(
        openapi_path=schema_file,
        database_url="sqlite+pysqlite:///:memory:",
        session_secret="test-session-secret",
        app_base_url="http://testserver",
        environment="test",
        version="9.9.9",
        stripe_price_starter="price_starter",
        stripe_price_pro="price_pro",
        stripe_webhook_secret="whsec_test",
        local_billing=True,
    )
    created = create_app(settings)
    created.state.oauth_providers = {"google": FakeProvider()}
    created.state.billing = FakeBilling()
    created.state.key_cache = KeyCache(fakeredis.FakeRedis(decode_responses=True))
    return created


@pytest.fixture
def client(app: Any) -> TestClient:
    return TestClient(app)


def _login(client: TestClient) -> None:
    response = client.get("/auth/login/google", follow_redirects=False)
    assert response.status_code == 303
    response = client.get(
        "/auth/callback/google?code=x&state=y", follow_redirects=False
    )
    assert response.status_code == 303
    assert response.headers["location"] == "/dashboard"


def _csrf(client: TestClient) -> str:
    response = client.get("/dashboard")
    assert response.status_code == 200
    match = CSRF_RE.search(response.text)
    assert match is not None
    return match.group(1)


def _activate(app: Any, plan: str = "starter", status: str = "active") -> None:
    with app.state.db.session_factory() as db:
        org = db.scalars(select(Organization)).first()
        assert org is not None
        org.plan = plan
        org.subscription_status = status
        org.stripe_customer_id = "cus_test"
        db.commit()


def _org_id(app: Any) -> Any:
    with app.state.db.session_factory() as db:
        return db.scalars(select(Organization.id)).first()


def _event(
    event_id: str,
    event_type: str,
    obj: dict[str, Any],
) -> bytes:
    return json.dumps(
        {"id": event_id, "type": event_type, "data": {"object": obj}}
    ).encode()


def test_protected_route_redirects_to_login(client: TestClient) -> None:
    response = client.get("/dashboard", follow_redirects=False)
    assert response.status_code == 303
    assert response.headers["location"] == "/login"


def test_login_page_links_providers(client: TestClient) -> None:
    response = client.get("/login")
    assert response.status_code == 200
    assert 'href="/auth/login/google"' in response.text
    assert 'href="/auth/login/github"' not in response.text
    assert 'href="/terms"' in response.text
    assert 'href="/privacy"' in response.text
    assert "mailto:" not in response.text
    assert "mrz-plate" not in response.text


def test_signed_in_navigation_follows_the_user_across_pages(client: TestClient) -> None:
    _login(client)
    token = _csrf(client)
    for path in ("/", "/dashboard", "/privacy", "/missing"):
        response = client.get(path)
        assert response.text.count('<header class="site-header">') == 1
        header = response.text.split("</header>", 1)[0]
        assert 'href="/dashboard"' in header
        assert 'href="/docs"' not in header
        assert 'href="/#pricing"' not in header
        assert 'href="/#faq"' not in header
        assert 'href="https://github.com/malvavisc0/blitzid"' in header
        assert "<span>GitHub</span>" in header
        assert 'action="/auth/logout"' in header
        assert f'name="csrf_token" value="{token}"' in header
        assert 'href="/login"' not in response.text
        assert response.headers["cache-control"] == "no-store"
        if path == "/dashboard":
            assert 'href="/dashboard" aria-current="page"' in header

    docs = client.get("/docs")
    assert docs.status_code == 200
    assert "/static/scalar.js" in docs.text
    assert '<header class="site-header">' not in docs.text
    assert 'class="skip-link"' not in docs.text

    assert 'href="/dashboard">Open dashboard</a>' in client.get("/").text
    login = client.get("/login", follow_redirects=False)
    assert login.headers["location"] == "/dashboard"
    home = client.post("/auth/logout", data={"csrf_token": token})
    assert 'href="/login"' in home.text
    assert 'action="/auth/logout"' not in home.text


def test_stale_session_does_not_redirect_to_dashboard(
    app: Any, client: TestClient
) -> None:
    _login(client)
    with app.state.db.session_factory() as db:
        db.delete(db.scalars(select(Membership)).one())
        db.commit()

    login = client.get("/login", follow_redirects=False)
    assert login.status_code == 200
    assert "Sign in to BlitzID" in login.text
    assert 'action="/auth/logout"' not in login.text


def test_oauth_callback_creates_one_user_and_org(app: Any, client: TestClient) -> None:
    _login(client)
    dashboard = client.get("/dashboard")
    assert dashboard.status_code == 200
    assert "Choose a plan" in dashboard.text
    with app.state.db.session_factory() as db:
        assert db.scalar(select(func.count()).select_from(User)) == 1
        assert db.scalar(select(func.count()).select_from(Organization)) == 1
        assert db.scalar(select(func.count()).select_from(Identity)) == 1

    # Signing in again links to the same account.
    client.get("/auth/login/google", follow_redirects=False)
    client.get("/auth/callback/google?code=x&state=y", follow_redirects=False)
    with app.state.db.session_factory() as db:
        assert db.scalar(select(func.count()).select_from(User)) == 1
        assert db.scalar(select(func.count()).select_from(Organization)) == 1


def test_oauth_failure_redirects_with_error(app: Any, client: TestClient) -> None:
    app.state.oauth_providers = {"google": BrokenProvider()}
    response = client.get(
        "/auth/callback/google?code=x&state=y", follow_redirects=False
    )
    assert response.status_code == 303
    assert response.headers["location"] == "/login?error=oauth"


def test_logout_requires_csrf_and_clears_session(app: Any, client: TestClient) -> None:
    _login(client)
    missing = client.post("/auth/logout", follow_redirects=False)
    assert missing.status_code == 403
    token = _csrf(client)
    response = client.post(
        "/auth/logout", data={"csrf_token": token}, follow_redirects=False
    )
    assert response.status_code == 303
    assert client.get("/dashboard", follow_redirects=False).status_code == 303


def test_no_plan_blocks_key_creation(app: Any, client: TestClient) -> None:
    _login(client)
    token = _csrf(client)
    response = client.post(
        "/dashboard/keys",
        data={"name": "Prod", "csrf_token": token},
        follow_redirects=False,
    )
    assert response.status_code == 303
    assert response.headers["location"] == "/dashboard?error=plan"
    with app.state.db.session_factory() as db:
        assert db.scalar(select(func.count()).select_from(ApiKey)) == 0


def test_create_key_shows_plaintext_once(app: Any, client: TestClient) -> None:
    _login(client)
    _activate(app, plan="starter")
    token = _csrf(client)
    response = client.post(
        "/dashboard/keys",
        data={"name": "Prod", "csrf_token": token},
    )
    assert response.status_code == 200
    assert "shown once" in response.text
    assert response.headers["cache-control"] == "no-store"
    match = KEY_RE.search(response.text)
    assert match is not None
    plaintext = match.group(0)
    with app.state.db.session_factory() as db:
        key = db.scalars(select(ApiKey)).one()
        assert key.secret_hash not in plaintext
        assert key.name == "Prod"
    dashboard = client.get("/dashboard")
    assert plaintext not in dashboard.text
    assert "blz_live_" in dashboard.text


def test_create_key_enforces_active_key_limit(app: Any, client: TestClient) -> None:
    _login(client)
    _activate(app, plan="starter")
    token = _csrf(client)
    first = client.post(
        "/dashboard/keys",
        data={"name": "Prod", "csrf_token": token},
    )
    assert first.status_code == 200
    second = client.post(
        "/dashboard/keys",
        data={"name": "Second", "csrf_token": token},
        follow_redirects=False,
    )
    assert second.status_code == 303
    assert second.headers["location"] == "/dashboard?error=keylimit"
    dashboard = client.get("/dashboard").text
    assert "1 of 1 active" in dashboard
    assert 'action="/dashboard/keys"' not in dashboard


def test_dashboard_shows_usage_without_an_empty_chart(
    app: Any, client: TestClient
) -> None:
    _login(client)
    _activate(app)
    token = _csrf(client)
    dashboard = client.get("/dashboard").text
    assert "Current plan" in dashboard
    assert 'href="/docs">API docs</a>' in dashboard
    assert "$5 / month" in dashboard
    assert "1,000 requests remaining" in dashboard
    assert "Quickstart" not in dashboard
    assert 'class="usage-chart"' not in dashboard

    client.post("/dashboard/keys", data={"csrf_token": token})
    with app.state.db.session_factory() as db:
        key = db.scalars(select(ApiKey)).one()
        db.add(
            UsageDaily(
                org_id=key.org_id, api_key_id=key.id, day=utcnow().date(), requests=42
            )
        )
        db.commit()
    dashboard = client.get("/dashboard").text
    assert 'class="usage-chart"' in dashboard
    assert "958 requests remaining" in dashboard
    assert 'class="usage-progress"' in dashboard

    with app.state.db.session_factory() as db:
        db.scalars(select(UsageDaily)).one().requests = 1_000
        db.commit()
    dashboard = client.get("/dashboard").text
    assert 'class="usage-progress usage-progress--warning"' in dashboard
    assert "Monthly quota reached." in dashboard


def test_rotate_and_revoke(app: Any, client: TestClient) -> None:
    _login(client)
    _activate(app, plan="starter")
    token = _csrf(client)
    client.post("/dashboard/keys", data={"name": "Prod", "csrf_token": token})
    assert 'button--danger button--small" type="submit">Revoke' in (
        client.get("/dashboard").text
    )
    with app.state.db.session_factory() as db:
        old_id = db.scalars(select(ApiKey.id)).one()
    rotated = client.post(
        f"/dashboard/keys/{old_id}/rotate", data={"csrf_token": token}
    )
    assert rotated.status_code == 200
    assert "Key rotated" in rotated.text
    assert (
        'class="status status--warning">Rotating</span>'
        in client.get("/dashboard").text
    )
    with app.state.db.session_factory() as db:
        keys = list(db.scalars(select(ApiKey)))
        assert len(keys) == 2
        old = db.get(ApiKey, old_id)
        assert old is not None and old.expires_at is not None

    new_id = next(k.id for k in keys if k.id != old_id)
    revoked = client.post(
        f"/dashboard/keys/{new_id}/revoke",
        data={"csrf_token": token},
        follow_redirects=False,
    )
    assert revoked.status_code == 303
    with app.state.db.session_factory() as db:
        key = db.get(ApiKey, new_id)
        assert key is not None and key.revoked_at is not None
    assert "Revoked" in client.get("/dashboard").text


def test_missing_csrf_is_rejected(app: Any, client: TestClient) -> None:
    _login(client)
    _activate(app)
    response = client.post("/dashboard/keys", data={"name": "Prod"})
    assert response.status_code == 403


def test_checkout_and_portal(app: Any, client: TestClient) -> None:
    _login(client)
    token = _csrf(client)
    checkout = client.post(
        "/billing/checkout",
        data={"plan": "pro", "csrf_token": token},
        follow_redirects=False,
    )
    assert checkout.status_code == 303
    assert checkout.headers["location"] == "https://checkout.test/session"
    assert app.state.billing.checkout_args["plan"] == "pro"
    assert app.state.billing.checkout_args["success"] == (
        "http://testserver/dashboard?checkout=success"
    )
    with app.state.db.session_factory() as db:
        org = db.scalars(select(Organization)).one()
        assert org.stripe_customer_id == "cus_test"

    portal = client.post(
        "/billing/portal", data={"csrf_token": token}, follow_redirects=False
    )
    assert portal.status_code == 303
    assert portal.headers["location"] == "https://portal.test/session"


def test_legacy_billing_return_waits_for_checkout_sync(client: TestClient) -> None:
    _login(client)
    redirect = client.get("/billing?checkout=success", follow_redirects=False)
    assert redirect.status_code == 303
    assert redirect.headers["location"] == "/dashboard?checkout=success"
    dashboard = client.get(redirect.headers["location"]).text
    assert "Stripe is confirming your subscription" in dashboard
    assert 'class="app-alert app-alert--warning" role="status"' in dashboard
    assert 'action="/billing/checkout"' not in dashboard


def test_plan_switch_uses_stripe_instead_of_checkout(
    app: Any, client: TestClient
) -> None:
    _login(client)
    _activate(app)
    token = _csrf(client)
    dashboard = client.get("/dashboard")
    assert 'href="/dashboard?plans=1#plan-options"' in dashboard.text
    assert "Manage billing" in dashboard.text
    compare = client.get("/dashboard?plans=1")
    assert "Compare plans" in compare.text
    assert "Switch to Pro" in compare.text
    assert "Switch to Starter" not in compare.text

    assert client.post("/billing/change-plan", data={"plan": "pro"}).status_code == 403
    checkout = client.post(
        "/billing/checkout",
        data={"plan": "pro", "csrf_token": token},
        follow_redirects=False,
    )
    assert checkout.headers["location"] == "/dashboard?error=already"
    change = client.post(
        "/billing/change-plan",
        data={"plan": "pro", "csrf_token": token},
        follow_redirects=False,
    )
    assert change.status_code == 303
    assert change.headers["location"] == "https://portal.test/confirm-plan"
    assert app.state.billing.plan_change_args == (
        "cus_test",
        "starter",
        "pro",
        "http://testserver/dashboard",
    )
    assert app.state.billing.checkout_args is None
    with app.state.db.session_factory() as db:
        assert db.scalars(select(Organization.plan)).one() == "starter"


def test_cancel_plan_opens_stripe_confirmation_without_disabling_access(
    app: Any, client: TestClient
) -> None:
    anonymous = client.post("/billing/cancel", follow_redirects=False)
    assert anonymous.headers["location"] == "/login"
    _login(client)
    token = _csrf(client)
    unplanned = client.post(
        "/billing/cancel", data={"csrf_token": token}, follow_redirects=False
    )
    assert unplanned.headers["location"] == "/dashboard?error=cancel-unavailable"

    _activate(app)
    dashboard = client.get("/dashboard").text
    assert (
        'action="/billing/cancel"' in dashboard and ">Cancel plan</button>" in dashboard
    )
    assert client.post("/billing/cancel").status_code == 403
    assert app.state.billing.cancel_args is None
    response = client.post(
        "/billing/cancel", data={"csrf_token": token}, follow_redirects=False
    )
    assert response.status_code == 303
    assert response.headers["location"] == "https://portal.test/confirm-cancellation"
    assert app.state.billing.cancel_args == ("cus_test", "http://testserver/dashboard")

    pending = client.get("/dashboard?billing=cancellation").text
    assert "Stripe is syncing your cancellation" in pending
    assert "Usage this month" in pending
    with app.state.db.session_factory() as db:
        org = db.scalars(select(Organization)).one()
        assert (org.plan, org.subscription_status, org.cancel_at) == (
            "starter",
            "active",
            None,
        )


def test_cancel_plan_handles_stripe_portal_unavailability(
    app: Any, client: TestClient
) -> None:
    _login(client)
    _activate(app)
    app.state.billing.create_cancellation = Mock(
        side_effect=stripe.InvalidRequestError("Cancellation disabled", None)
    )
    response = client.post("/billing/cancel", data={"csrf_token": _csrf(client)})
    assert "Could not open Stripe cancellation. Try Manage billing" in response.text
    assert "Usage this month" in response.text


def test_stripe_checkout_collects_company_invoice_details(
    app: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    create = Mock(return_value={"url": "https://checkout.test/session"})
    monkeypatch.setattr(stripe, "api_key", None)
    monkeypatch.setattr(stripe.checkout.Session, "create", create)
    billing = StripeBilling(
        replace(app.state.settings, stripe_secret_key="sk_test_fake")
    )

    assert (
        billing.create_checkout(
            "cus_test",
            "pro",
            "org_test",
            "https://blitzid.io/dashboard?checkout=success",
            "https://blitzid.io/dashboard?checkout=cancelled",
        )
        == "https://checkout.test/session"
    )
    create.assert_called_once()
    params = create.call_args.kwargs
    assert params["mode"] == "subscription"
    assert params["customer"] == "cus_test"
    assert params["billing_address_collection"] == "required"
    assert params["name_collection"] == {
        "business": {"enabled": True, "optional": True}
    }
    assert params["tax_id_collection"] == {"enabled": True}
    assert params["customer_update"] == {"address": "auto", "name": "auto"}
    assert params["automatic_tax"] == {"enabled": True}


def test_stripe_cancellation_targets_one_existing_subscription(
    app: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    subscription = {"id": "sub_test", "status": "past_due"}
    subscriptions = SimpleNamespace(data=[subscription], has_more=False)
    list_subscriptions = Mock(return_value=subscriptions)
    portal = Mock(return_value={"url": "https://portal.test/confirm-cancellation"})
    monkeypatch.setattr(stripe, "api_key", None)
    monkeypatch.setattr(stripe.Subscription, "list", list_subscriptions)
    monkeypatch.setattr(stripe.billing_portal.Session, "create", portal)
    billing = StripeBilling(
        replace(app.state.settings, stripe_secret_key="sk_test_fake")
    )

    return_url = "https://blitzid.io/dashboard"
    assert (
        billing.create_cancellation("cus_test", return_url)
        == portal.return_value["url"]
    )
    list_subscriptions.assert_called_once_with(customer="cus_test", limit=2)
    portal.assert_called_once_with(
        customer="cus_test",
        return_url=return_url,
        flow_data={
            "type": "subscription_cancel",
            "subscription_cancel": {"subscription": "sub_test"},
            "after_completion": {
                "type": "redirect",
                "redirect": {"return_url": f"{return_url}?billing=cancellation"},
            },
        },
    )
    subscriptions.data.append({"id": "sub_other", "status": "active"})
    with pytest.raises(BillingError, match="Expected one open Stripe subscription"):
        billing.create_cancellation("cus_test", return_url)
    assert portal.call_count == 1


def test_stripe_plan_change_confirms_a_matching_subscription_item(
    app: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    item = {"id": "si_test", "price": {"id": "price_starter"}}
    subscription = {
        "id": "sub_test",
        "status": "active",
        "cancel_at": None,
        "cancel_at_period_end": False,
        "items": {"has_more": False, "data": [item]},
    }
    list_subscriptions = Mock(
        return_value=SimpleNamespace(data=[subscription], has_more=False)
    )
    portal = Mock(return_value={"url": "https://portal.test/confirm-plan"})
    monkeypatch.setattr(stripe, "api_key", None)
    monkeypatch.setattr(stripe.Subscription, "list", list_subscriptions)
    monkeypatch.setattr(stripe.billing_portal.Session, "create", portal)
    billing = StripeBilling(
        replace(app.state.settings, stripe_secret_key="sk_test_fake")
    )

    assert billing.create_plan_change(
        "cus_test", "starter", "pro", "https://app.test"
    ) == ("https://portal.test/confirm-plan")
    list_subscriptions.assert_called_once_with(customer="cus_test", limit=2)
    portal.assert_called_once_with(
        customer="cus_test",
        return_url="https://app.test",
        flow_data={
            "type": "subscription_update_confirm",
            "subscription_update_confirm": {
                "subscription": "sub_test",
                "items": [{"id": "si_test", "price": "price_pro"}],
            },
        },
    )

    item["price"]["id"] = "price_pro"
    with pytest.raises(BillingError, match="doesn't match the current plan"):
        billing.create_plan_change("cus_test", "starter", "pro", "https://app.test")
    assert portal.call_count == 1


def test_past_due_account_shows_payment_recovery(app: Any, client: TestClient) -> None:
    _login(client)
    _activate(app, status="past_due")
    response = client.get("/dashboard?plans=1")
    assert 'class="status status--warning">past due</span>' in response.text
    assert "Update your payment method to restore API access" in response.text
    assert "Manage billing" in response.text
    assert 'action="/billing/cancel"' in response.text
    assert "Change plan" not in response.text
    assert "Choose a plan" not in response.text
    assert "Usage this month" not in response.text


def test_plan_selection_activates_locally_without_stripe(
    app: Any, client: TestClient
) -> None:
    app.state.billing = NullBilling()
    _login(client)
    token = _csrf(client)
    checkout = client.post(
        "/billing/checkout",
        data={"plan": "starter", "csrf_token": token},
        follow_redirects=False,
    )
    assert checkout.status_code == 303
    assert checkout.headers["location"] == "/dashboard?checkout=local"
    dashboard = client.get("/dashboard?checkout=local")
    assert "Plan activated locally" in dashboard.text
    with app.state.db.session_factory() as db:
        org = db.scalars(select(Organization)).one()
        assert (org.plan, org.subscription_status) == ("starter", "active")
        assert org.stripe_customer_id is None


def test_local_billing_requires_explicit_opt_in(app: Any, client: TestClient) -> None:
    app.state.billing = NullBilling()
    app.state.settings = replace(app.state.settings, local_billing=False)
    _login(client)
    token = _csrf(client)
    checkout = client.post(
        "/billing/checkout",
        data={"plan": "starter", "csrf_token": token},
        follow_redirects=False,
    )
    assert checkout.headers["location"] == "/dashboard?error=checkout"


def test_local_plan_switch_keeps_billing_period_and_existing_keys(
    app: Any, client: TestClient
) -> None:
    app.state.billing = NullBilling()
    _login(client)
    token = _csrf(client)
    client.post("/billing/checkout", data={"plan": "pro", "csrf_token": token})
    for name in ("Production", "Staging"):
        created = client.post(
            "/dashboard/keys", data={"name": name, "csrf_token": token}
        )
        assert created.status_code == 200
    with app.state.db.session_factory() as db:
        period_end = db.scalars(select(Organization.current_period_end)).one()

    for target in ("starter", "pro"):
        switched = client.post(
            "/billing/change-plan", data={"plan": target, "csrf_token": token}
        )
        assert "Plan changed locally" in switched.text
        assert f"<strong>{target.capitalize()}</strong>" in switched.text
        if target == "starter":
            assert "2 of 1 active" in switched.text
            assert 'action="/dashboard/keys"' not in switched.text
        with app.state.db.session_factory() as db:
            org = db.scalars(select(Organization)).one()
            assert (org.plan, org.current_period_end) == (target, period_end)
            assert db.scalar(select(func.count()).select_from(ApiKey)) == 2

    app.state.settings = replace(app.state.settings, environment="production")
    disabled = client.post(
        "/billing/change-plan",
        data={"plan": "starter", "csrf_token": token},
        follow_redirects=False,
    )
    assert disabled.headers["location"] == "/dashboard?error=change-unavailable"


def test_preview_cancel_requires_confirmation_and_keeps_account_and_keys(
    app: Any, client: TestClient
) -> None:
    app.state.billing = NullBilling()
    _login(client)
    data = {"csrf_token": _csrf(client)}
    dashboard = client.post("/billing/checkout", data={**data, "plan": "pro"})
    assert 'href="/dashboard?cancel=1#cancel-plan">Cancel plan</a>' in dashboard.text
    assert "Manage billing" not in dashboard.text
    assert client.post("/dashboard/keys", data=data).status_code == 200

    confirmation = client.post("/billing/cancel", data=data)
    assert "Cancel preview plan" in confirmation.text
    assert "Keep plan" in confirmation.text
    assert "Usage this month" in confirmation.text

    development_settings = app.state.settings
    app.state.settings = replace(development_settings, environment="production")
    data["confirm_preview"] = "true"
    unavailable = client.post("/billing/cancel", data=data, follow_redirects=False)
    assert unavailable.headers["location"] == "/dashboard?error=cancel-unavailable"
    app.state.settings = development_settings

    canceled = client.post("/billing/cancel", data=data)
    assert "Preview plan canceled" in canceled.text
    assert "Choose a plan" in canceled.text
    assert "Usage this month" not in canceled.text
    with app.state.db.session_factory() as db:
        org = db.scalars(select(Organization)).one()
        assert (org.plan, org.subscription_status, org.current_period_end) == (
            "none",
            "canceled",
            None,
        )
        assert db.scalar(select(func.count()).select_from(ApiKey)) == 1
        assert db.scalar(select(func.count()).select_from(User)) == 1


def test_plan_selection_still_fails_without_stripe_in_production(
    app: Any, client: TestClient
) -> None:
    app.state.settings = replace(app.state.settings, environment="production")
    app.state.billing = NullBilling()
    _login(client)
    token = _csrf(client)
    checkout = client.post(
        "/billing/checkout",
        data={"plan": "starter", "csrf_token": token},
        follow_redirects=False,
    )
    assert checkout.status_code == 303
    assert checkout.headers["location"] == "/dashboard?error=checkout"


def test_webhook_activates_and_is_idempotent(app: Any, client: TestClient) -> None:
    _login(client)
    org_id = _org_id(app)
    payload = _event(
        "evt_1",
        "checkout.session.completed",
        {
            "client_reference_id": str(org_id),
            "customer": "cus_test",
            "subscription": "sub_1",
        },
    )
    response = client.post(
        "/webhooks/stripe",
        content=payload,
        headers={"stripe-signature": "test-signature"},
    )
    assert response.status_code == 200
    with app.state.db.session_factory() as db:
        org = db.get(Organization, org_id)
        assert org is not None
        assert (org.plan, org.subscription_status) == ("starter", "active")
        assert db.scalar(select(func.count()).select_from(StripeEvent)) == 1

    duplicate = client.post(
        "/webhooks/stripe",
        content=payload,
        headers={"stripe-signature": "test-signature"},
    )
    assert duplicate.status_code == 200
    with app.state.db.session_factory() as db:
        assert db.scalar(select(func.count()).select_from(StripeEvent)) == 1


@pytest.mark.parametrize("flexible", [False, True])
def test_webhook_syncs_scheduled_cancellation_and_plan_end(
    app: Any, client: TestClient, flexible: bool
) -> None:
    _login(client)
    _activate(app)
    token = _csrf(client)
    assert client.post("/dashboard/keys", data={"csrf_token": token}).status_code == 200
    end = (utcnow() + timedelta(days=7)).replace(microsecond=0)
    subscription = {
        "customer": "cus_test",
        "status": "active",
        "items": {
            "data": [
                {
                    "price": {"id": "price_starter"},
                    "current_period_end": end.timestamp(),
                }
            ]
        },
    }

    def notify(event_id: str, kind: str = "updated", **fields: Any) -> None:
        response = client.post(
            "/webhooks/stripe",
            content=_event(
                event_id, f"customer.subscription.{kind}", {**subscription, **fields}
            ),
            headers={"stripe-signature": "test-signature"},
        )
        assert response.status_code == 200

    scheduled = (
        {"cancel_at": end.timestamp()} if flexible else {"cancel_at_period_end": True}
    )
    notify("evt_scheduled", **scheduled)
    dashboard = client.get("/dashboard?billing=cancellation").text
    assert 'class="status status--warning">Ending</span>' in dashboard
    assert f'datetime="{end.isoformat()}"' in dashboard
    assert "Your plan won't renew" in dashboard
    assert "Stripe is syncing" not in dashboard
    assert "Usage this month" in dashboard
    assert "Manage billing" in dashboard
    assert "Cancel plan" not in dashboard and "Change plan" not in dashboard

    notify("evt_resumed", cancel_at=None, cancel_at_period_end=False)
    assert ">Cancel plan</button>" in client.get("/dashboard").text

    notify("evt_expired", cancel_at=(utcnow() - timedelta(minutes=1)).timestamp())
    expired = client.get("/dashboard").text
    assert "API access has stopped" in expired
    assert "Usage this month" not in expired
    assert 'action="/billing/checkout"' not in expired

    notify("evt_deleted", "deleted", status="canceled")
    final_invoice_failure = client.post(
        "/webhooks/stripe",
        content=_event(
            "evt_final_invoice", "invoice.payment_failed", {"customer": "cus_test"}
        ),
        headers={"stripe-signature": "test-signature"},
    )
    assert final_invoice_failure.status_code == 200
    ended = client.get("/dashboard").text
    assert "Your previous plan has ended" in ended
    assert "Manage billing" in ended and "Choose a plan" in ended
    with app.state.db.session_factory() as db:
        org = db.scalars(select(Organization)).one()
        assert (org.plan, org.subscription_status, org.cancel_at) == (
            "none",
            "canceled",
            None,
        )
        assert org.stripe_customer_id == "cus_test"
        assert db.scalar(select(func.count()).select_from(ApiKey)) == 1


def test_webhook_subscription_and_invoice_updates(app: Any, client: TestClient) -> None:
    _login(client)
    org_id = _org_id(app)
    _activate(app)
    subscription = _event(
        "evt_2",
        "customer.subscription.updated",
        {
            "customer": "cus_test",
            "status": "active",
            "items": {
                "data": [
                    {"price": {"id": "price_pro"}, "current_period_end": 1798761600}
                ]
            },
        },
    )
    assert (
        client.post(
            "/webhooks/stripe",
            content=subscription,
            headers={"stripe-signature": "test-signature"},
        ).status_code
        == 200
    )
    with app.state.db.session_factory() as db:
        org = db.get(Organization, org_id)
        assert org is not None and org.plan == "pro"
    dashboard = client.get("/dashboard").text
    assert "$19 / month" in dashboard and "10,000 requests remaining" in dashboard
    assert 'datetime="2027-01-01"' in dashboard

    failed = _event(
        "evt_3",
        "invoice.payment_failed",
        {"customer": "cus_test"},
    )
    client.post(
        "/webhooks/stripe",
        content=failed,
        headers={"stripe-signature": "test-signature"},
    )
    with app.state.db.session_factory() as db:
        org = db.get(Organization, org_id)
        assert org is not None and org.grace_until is not None

    paid = _event("evt_4", "invoice.paid", {"customer": "cus_test"})
    client.post(
        "/webhooks/stripe",
        content=paid,
        headers={"stripe-signature": "test-signature"},
    )
    with app.state.db.session_factory() as db:
        org = db.get(Organization, org_id)
        assert org is not None and org.grace_until is None


def test_webhook_rejects_bad_signature(client: TestClient) -> None:
    response = client.post(
        "/webhooks/stripe",
        content=_event("evt_x", "invoice.paid", {"customer": "cus_test"}),
        headers={"stripe-signature": "wrong"},
    )
    assert response.status_code == 400
