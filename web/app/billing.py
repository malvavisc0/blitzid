"""Stripe gateway, entitlement rules, and plan/price mapping."""

from __future__ import annotations

import json
import uuid
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from typing import Any, Protocol, cast

import stripe
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.config import Settings
from app.entitlement import ACTIVE_STATUSES, entitled
from app.models import Organization, aware_utc
from app.plans import PLANS, SELF_SERVE_PLANS

GRACE_PERIOD = timedelta(days=3)
ACTIVE_SUBSCRIPTION_STATUSES = ACTIVE_STATUSES
CHECKOUT_BLOCKING_STATUSES = ACTIVE_STATUSES | {"past_due"}


class BillingUnavailable(RuntimeError):
    """Billing is not configured or Stripe is unreachable."""


class BillingError(RuntimeError):
    """A checkout/portal request cannot be fulfilled."""


@dataclass(frozen=True)
class SubscriptionSnapshot:
    status: str
    price_id: str | None
    current_period_end: datetime | None
    customer_id: str
    cancel_at: datetime | None = None


@dataclass(frozen=True)
class BillingEvent:
    id: str
    type: str
    object: dict[str, Any]


def is_entitled(org: Organization, now: datetime | None = None) -> bool:
    """Whether the organization may create/use API keys right now."""
    now = now or datetime.now(UTC)
    if org.plan not in PLANS:
        return False
    return entitled(
        org.subscription_status,
        aware_utc(org.cancel_at),
        aware_utc(org.grace_until),
        now,
    )


def plan_for_price(price_id: str | None, settings: Settings) -> str | None:
    """Map a Stripe price back to a plan name."""
    if price_id and price_id == settings.stripe_price_starter:
        return "starter"
    if price_id and price_id == settings.stripe_price_pro:
        return "pro"
    return None


def price_for_plan(plan: str, settings: Settings) -> str:
    """Map a self-serve plan to its configured Stripe price."""
    prices = {
        "starter": settings.stripe_price_starter,
        "pro": settings.stripe_price_pro,
    }
    price = prices.get(plan)
    if not price:
        raise BillingError(f"Stripe price is not configured for {plan}")
    return price


class BillingGateway(Protocol):
    """The operations the web app needs from Stripe."""

    def ensure_customer(self, org_id: str, email: str, name: str | None) -> str: ...

    def create_checkout(
        self,
        customer_id: str,
        plan: str,
        org_id: str,
        success_url: str,
        cancel_url: str,
    ) -> str: ...

    def create_portal(self, customer_id: str, return_url: str) -> str: ...

    def create_plan_change(
        self, customer_id: str, current_plan: str, plan: str, return_url: str
    ) -> str: ...

    def create_cancellation(self, customer_id: str, return_url: str) -> str: ...

    def parse_event(self, payload: bytes, signature: str) -> BillingEvent: ...

    def fetch_subscription(
        self, subscription_id: str
    ) -> SubscriptionSnapshot | None: ...


class NullBilling:
    """Used when Stripe is not configured (local dev, self-hosted web)."""

    def ensure_customer(self, org_id: str, email: str, name: str | None) -> str:
        raise BillingUnavailable("Stripe is not configured")

    def create_checkout(
        self,
        customer_id: str,
        plan: str,
        org_id: str,
        success_url: str,
        cancel_url: str,
    ) -> str:
        raise BillingUnavailable("Stripe is not configured")

    def create_portal(self, customer_id: str, return_url: str) -> str:
        raise BillingUnavailable("Stripe is not configured")

    def create_plan_change(
        self, customer_id: str, current_plan: str, plan: str, return_url: str
    ) -> str:
        raise BillingUnavailable("Stripe is not configured")

    def create_cancellation(self, customer_id: str, return_url: str) -> str:
        raise BillingUnavailable("Stripe is not configured")

    def parse_event(self, payload: bytes, signature: str) -> BillingEvent:
        raise BillingUnavailable("Stripe is not configured")

    def fetch_subscription(self, subscription_id: str) -> SubscriptionSnapshot | None:
        raise BillingUnavailable("Stripe is not configured")


def local_billing_enabled(
    org: Organization, gateway: BillingGateway, settings: Settings
) -> bool:
    return (
        settings.local_billing
        and settings.environment != "production"
        and isinstance(gateway, NullBilling)
        and not org.stripe_customer_id
    )


def can_manage_billing(org: Organization, gateway: BillingGateway) -> bool:
    return bool(org.stripe_customer_id) and not isinstance(gateway, NullBilling)


def can_cancel_plan(
    org: Organization, gateway: BillingGateway, settings: Settings
) -> bool:
    return (
        org.plan in SELF_SERVE_PLANS
        and org.subscription_status in CHECKOUT_BLOCKING_STATUSES
        and org.cancel_at is None
        and (
            can_manage_billing(org, gateway)
            or local_billing_enabled(org, gateway, settings)
        )
    )


def can_switch_plan(
    org: Organization, gateway: BillingGateway, settings: Settings
) -> bool:
    if (
        org.plan not in SELF_SERVE_PLANS
        or org.subscription_status not in ACTIVE_SUBSCRIPTION_STATUSES
        or org.cancel_at is not None
    ):
        return False
    if org.stripe_customer_id:
        return can_manage_billing(org, gateway)
    return local_billing_enabled(org, gateway, settings)


class StripeBilling:
    """Thin Stripe SDK adapter."""

    def __init__(self, settings: Settings) -> None:
        if not settings.stripe_secret_key:
            raise BillingUnavailable("STRIPE_SECRET_KEY is not set")
        stripe.api_key = settings.stripe_secret_key
        self._settings = settings

    def ensure_customer(self, org_id: str, email: str, name: str | None) -> str:
        params: dict[str, Any] = {
            "email": email,
            "metadata": {"org_id": org_id},
        }
        if name:
            params["name"] = name
        customer = stripe.Customer.create(**params)
        return str(customer["id"])

    def create_checkout(
        self,
        customer_id: str,
        plan: str,
        org_id: str,
        success_url: str,
        cancel_url: str,
    ) -> str:
        session = stripe.checkout.Session.create(
            mode="subscription",
            customer=customer_id,
            client_reference_id=org_id,
            line_items=[{"price": price_for_plan(plan, self._settings), "quantity": 1}],
            success_url=success_url,
            cancel_url=cancel_url,
            allow_promotion_codes=False,
            automatic_tax={"enabled": True},
            billing_address_collection="required",
            name_collection={"business": {"enabled": True, "optional": True}},
            tax_id_collection={"enabled": True},
            customer_update={"address": "auto", "name": "auto"},
            metadata={"org_id": org_id},
            subscription_data={"metadata": {"org_id": org_id}},
        )
        url = cast(Any, session).get("url")
        if not url:
            raise BillingError("Stripe returned no checkout URL")
        return str(url)

    def create_portal(self, customer_id: str, return_url: str) -> str:
        session = stripe.billing_portal.Session.create(
            customer=customer_id, return_url=return_url
        )
        return str(session["url"])

    @staticmethod
    def _open_subscription(customer_id: str) -> stripe.Subscription:
        # Never guess which subscription to change if the customer has more than one.
        subscriptions = stripe.Subscription.list(customer=customer_id, limit=2)
        if subscriptions.has_more or len(subscriptions.data) != 1:
            raise BillingError("Expected one open Stripe subscription")
        return subscriptions.data[0]

    def create_cancellation(self, customer_id: str, return_url: str) -> str:
        subscription = self._open_subscription(customer_id)
        if subscription["status"] not in CHECKOUT_BLOCKING_STATUSES:
            raise BillingError("No cancellable Stripe subscription")
        session = stripe.billing_portal.Session.create(
            customer=customer_id,
            return_url=return_url,
            flow_data={
                "type": "subscription_cancel",
                "subscription_cancel": {"subscription": subscription["id"]},
                "after_completion": {
                    "type": "redirect",
                    "redirect": {"return_url": f"{return_url}?billing=cancellation"},
                },
            },
        )
        return str(session["url"])

    def create_plan_change(
        self, customer_id: str, current_plan: str, plan: str, return_url: str
    ) -> str:
        """Send the customer to Stripe to confirm an update, not a new checkout."""
        current_price = price_for_plan(current_plan, self._settings)
        target_price = price_for_plan(plan, self._settings)
        if current_price == target_price:
            raise BillingError("The current and target Stripe prices match")

        subscription = self._open_subscription(customer_id)
        if (
            subscription["cancel_at"] is not None
            or subscription["cancel_at_period_end"]
        ):
            raise BillingError("Subscription is scheduled to end")
        items = subscription["items"]
        if (
            subscription["status"] not in ACTIVE_SUBSCRIPTION_STATUSES
            or items["has_more"]
            or len(items["data"]) != 1
            or items["data"][0]["price"]["id"] != current_price
        ):
            raise BillingError("The Stripe subscription doesn't match the current plan")

        session = stripe.billing_portal.Session.create(
            customer=customer_id,
            return_url=return_url,
            flow_data={
                "type": "subscription_update_confirm",
                "subscription_update_confirm": {
                    "subscription": subscription["id"],
                    "items": [{"id": items["data"][0]["id"], "price": target_price}],
                },
            },
        )
        return str(session["url"])

    def parse_event(self, payload: bytes, signature: str) -> BillingEvent:
        if not self._settings.stripe_webhook_secret:
            raise BillingUnavailable("STRIPE_WEBHOOK_SECRET is not set")
        stripe.Webhook.construct_event(
            payload, signature, self._settings.stripe_webhook_secret
        )
        data = json.loads(payload.decode("utf-8"))
        return BillingEvent(
            id=str(data["id"]),
            type=str(data["type"]),
            object=dict(data["data"]["object"]),
        )

    def fetch_subscription(self, subscription_id: str) -> SubscriptionSnapshot | None:
        subscription = stripe.Subscription.retrieve(subscription_id)
        return snapshot_from_object(subscription.to_dict())


def apply_subscription(
    org: Organization, snapshot: SubscriptionSnapshot, settings: Settings
) -> None:
    """Copy a Stripe subscription snapshot onto the local organization."""
    plan = plan_for_price(snapshot.price_id, settings)
    org.subscription_status = snapshot.status
    org.current_period_end = snapshot.current_period_end
    org.cancel_at = snapshot.cancel_at
    if plan is not None:
        org.plan = plan
    if snapshot.status in {"canceled", "unpaid", "incomplete_expired"}:
        org.plan = "none"
        org.cancel_at = None
        org.grace_until = None
    if snapshot.status == "past_due" and org.grace_until is None:
        org.grace_until = datetime.now(UTC) + GRACE_PERIOD
    elif snapshot.status in ACTIVE_SUBSCRIPTION_STATUSES:
        org.grace_until = None


def _first_item(obj: dict[str, Any]) -> dict[str, Any]:
    items = (obj.get("items") or {}).get("data") or []
    return items[0] if items else {}


def _timestamp(value: Any) -> datetime | None:
    return datetime.fromtimestamp(int(value), UTC) if value else None


def snapshot_from_object(obj: dict[str, Any]) -> SubscriptionSnapshot:
    """Build a snapshot from a Stripe subscription event object."""
    item = _first_item(obj)
    price_id = (item.get("price") or {}).get("id")
    period = item.get("current_period_end") or obj.get("current_period_end")
    # Flexible billing sets cancel_at; classic billing may use the period-end flag.
    cancel_at = obj.get("cancel_at")
    if cancel_at is None and obj.get("cancel_at_period_end"):
        cancel_at = period
    return SubscriptionSnapshot(
        status=str(obj.get("status", "inactive")),
        price_id=str(price_id) if price_id else None,
        current_period_end=_timestamp(period),
        customer_id=str(obj.get("customer", "")),
        cancel_at=_timestamp(cancel_at),
    )


def apply_event(
    db: Session,
    event: BillingEvent,
    gateway: BillingGateway,
    settings: Settings,
) -> Organization | None:
    """Apply one Stripe event to the local organization snapshot."""
    if event.type == "checkout.session.completed":
        return _apply_checkout_completed(db, event.object, gateway, settings)
    org = _org_for_customer(db, event.object.get("customer"))
    if org is None:
        return None
    return _apply_lifecycle_event(org, event, settings)


def _apply_checkout_completed(
    db: Session,
    obj: dict[str, Any],
    gateway: BillingGateway,
    settings: Settings,
) -> Organization | None:
    """Attach the customer and subscription created by a completed checkout."""
    raw_org = obj.get("client_reference_id") or (obj.get("metadata") or {}).get(
        "org_id"
    )
    try:
        org_id = uuid.UUID(str(raw_org))
    except (TypeError, ValueError):
        return None
    org = db.get(Organization, org_id)
    if org is None:
        return None
    customer = obj.get("customer")
    if customer:
        org.stripe_customer_id = str(customer)
    subscription_id = obj.get("subscription")
    if subscription_id:
        snapshot = gateway.fetch_subscription(str(subscription_id))
        if snapshot is not None:
            apply_subscription(org, snapshot, settings)
            if snapshot.customer_id:
                org.stripe_customer_id = snapshot.customer_id
    return org


def _org_for_customer(db: Session, customer: object) -> Organization | None:
    """Look up the organization that owns a Stripe customer id."""
    if not customer:
        return None
    return db.scalar(
        select(Organization).where(Organization.stripe_customer_id == str(customer))
    )


def _apply_lifecycle_event(
    org: Organization, event: BillingEvent, settings: Settings
) -> Organization | None:
    """Apply a subscription or invoice event to the organization."""
    if event.type in {
        "customer.subscription.created",
        "customer.subscription.updated",
        "customer.subscription.deleted",
    }:
        apply_subscription(org, snapshot_from_object(event.object), settings)
    elif event.type == "invoice.paid":
        org.grace_until = None
    elif event.type == "invoice.payment_failed":
        if org.plan not in PLANS:
            return None
        org.grace_until = datetime.now(UTC) + GRACE_PERIOD
        if org.subscription_status not in ACTIVE_SUBSCRIPTION_STATUSES:
            org.subscription_status = "past_due"
    else:
        return None
    return org
