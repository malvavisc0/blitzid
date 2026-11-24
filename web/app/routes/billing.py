"""Stripe Checkout, subscription changes, and the Customer Portal."""

from __future__ import annotations

import logging
from datetime import timedelta
from typing import Annotated

import stripe
from fastapi import APIRouter, Depends, Form, Request
from fastapi.responses import RedirectResponse
from sqlalchemy.orm import Session

from app import auth
from app.billing import (
    CHECKOUT_BLOCKING_STATUSES,
    BillingError,
    BillingUnavailable,
    can_cancel_plan,
    can_switch_plan,
    local_billing_enabled,
)
from app.db import get_db
from app.models import utcnow
from app.plans import SELF_SERVE_PLANS

_LOG = logging.getLogger(__name__)

router = APIRouter()


@router.get("/billing")
def billing_page(
    request: Request, db: Annotated[Session, Depends(get_db)]
) -> RedirectResponse:
    """Keep old bookmarks and outstanding Stripe return URLs working."""
    auth.current_session(request, db)
    query = request.url.query
    return RedirectResponse(
        f"/dashboard?{query}" if query else "/dashboard", status_code=303
    )


@router.post("/billing/checkout")
def start_checkout(
    request: Request,
    db: Annotated[Session, Depends(get_db)],
    plan: Annotated[str, Form()] = "",
    csrf_token: Annotated[str, Form()] = "",
) -> RedirectResponse:
    ctx = auth.current_session(request, db)
    auth.check_csrf(request, csrf_token)
    if plan not in SELF_SERVE_PLANS:
        return RedirectResponse("/dashboard?error=choose-plan", status_code=303)
    if ctx.org.subscription_status in CHECKOUT_BLOCKING_STATUSES:
        return RedirectResponse("/dashboard?error=already", status_code=303)
    gateway = request.app.state.billing
    settings = request.app.state.settings
    if local_billing_enabled(ctx.org, gateway, settings):
        ctx.org.plan = plan
        ctx.org.subscription_status = "active"
        ctx.org.current_period_end = utcnow() + timedelta(days=30)
        ctx.org.cancel_at = None
        db.commit()
        request.app.state.key_cache.invalidate_org(str(ctx.org.id))
        return RedirectResponse("/dashboard?checkout=local", status_code=303)
    try:
        customer_id = ctx.org.stripe_customer_id
        if not customer_id:
            customer_id = gateway.ensure_customer(
                str(ctx.org.id), ctx.user.email, ctx.user.name
            )
            ctx.org.stripe_customer_id = customer_id
            db.commit()
        url = gateway.create_checkout(
            customer_id,
            plan,
            str(ctx.org.id),
            f"{settings.app_base_url}/dashboard?checkout=success",
            f"{settings.app_base_url}/dashboard?checkout=cancelled",
        )
    except (BillingUnavailable, BillingError, stripe.StripeError):
        _LOG.exception("Checkout failed")
        return RedirectResponse("/dashboard?error=checkout", status_code=303)
    return RedirectResponse(url, status_code=303)


@router.post("/billing/change-plan")
def change_plan(
    request: Request,
    db: Annotated[Session, Depends(get_db)],
    plan: Annotated[str, Form()] = "",
    csrf_token: Annotated[str, Form()] = "",
) -> RedirectResponse:
    ctx = auth.current_session(request, db)
    auth.check_csrf(request, csrf_token)
    gateway = request.app.state.billing
    settings = request.app.state.settings
    if not can_switch_plan(ctx.org, gateway, settings):
        return RedirectResponse("/dashboard?error=change-unavailable", status_code=303)
    if plan not in SELF_SERVE_PLANS or plan == ctx.org.plan:
        return RedirectResponse(
            "/dashboard?plans=1&error=choose-other", status_code=303
        )

    customer_id = ctx.org.stripe_customer_id
    if not customer_id:
        # can_switch_plan allows this only for local, non-production accounts.
        ctx.org.plan = plan
        db.commit()
        request.app.state.key_cache.invalidate_org(str(ctx.org.id))
        return RedirectResponse("/dashboard?checkout=local-change", status_code=303)

    try:
        url = gateway.create_plan_change(
            customer_id, ctx.org.plan, plan, f"{settings.app_base_url}/dashboard"
        )
    except (BillingUnavailable, BillingError, stripe.StripeError):
        _LOG.exception("Plan change session failed")
        return RedirectResponse("/dashboard?plans=1&error=change-plan", status_code=303)
    return RedirectResponse(url, status_code=303)


@router.post("/billing/cancel")
def cancel_plan(
    request: Request,
    db: Annotated[Session, Depends(get_db)],
    csrf_token: Annotated[str, Form()] = "",
    confirm_preview: Annotated[bool, Form()] = False,
) -> RedirectResponse:
    ctx = auth.current_session(request, db)
    auth.check_csrf(request, csrf_token)
    gateway = request.app.state.billing
    settings = request.app.state.settings
    if not can_cancel_plan(ctx.org, gateway, settings):
        return RedirectResponse("/dashboard?error=cancel-unavailable", status_code=303)

    customer_id = ctx.org.stripe_customer_id
    if not customer_id:
        # Only a local, non-production preview can take this branch.
        if not confirm_preview:
            return RedirectResponse("/dashboard?cancel=1#cancel-plan", status_code=303)
        ctx.org.plan = "none"
        ctx.org.subscription_status = "canceled"
        ctx.org.current_period_end = None
        ctx.org.grace_until = None
        db.commit()
        request.app.state.key_cache.invalidate_org(str(ctx.org.id))
        return RedirectResponse("/dashboard?billing=preview-canceled", status_code=303)

    try:
        url = gateway.create_cancellation(
            customer_id, f"{settings.app_base_url}/dashboard"
        )
    except (BillingUnavailable, BillingError, stripe.StripeError):
        _LOG.exception("Cancellation session failed")
        return RedirectResponse("/dashboard?error=cancel-plan", status_code=303)
    return RedirectResponse(url, status_code=303)


@router.post("/billing/portal")
def open_portal(
    request: Request,
    db: Annotated[Session, Depends(get_db)],
    csrf_token: Annotated[str, Form()] = "",
) -> RedirectResponse:
    ctx = auth.current_session(request, db)
    auth.check_csrf(request, csrf_token)
    if not ctx.org.stripe_customer_id:
        return RedirectResponse("/dashboard?error=portal", status_code=303)
    try:
        url = request.app.state.billing.create_portal(
            ctx.org.stripe_customer_id,
            f"{request.app.state.settings.app_base_url}/dashboard",
        )
    except (BillingUnavailable, BillingError, stripe.StripeError):
        _LOG.exception("Portal failed")
        return RedirectResponse("/dashboard?error=portal", status_code=303)
    return RedirectResponse(url, status_code=303)
