"""Dashboard: monthly usage, API keys, and subscription management."""

from __future__ import annotations

import uuid
from collections.abc import Mapping
from datetime import date, timedelta
from typing import Annotated, Any

from fastapi import APIRouter, Depends, Form, Request
from fastapi.responses import HTMLResponse, RedirectResponse, Response
from sqlalchemy import func, select
from sqlalchemy.orm import Session

from app import audit, auth
from app.billing import (
    ACTIVE_SUBSCRIPTION_STATUSES,
    CHECKOUT_BLOCKING_STATUSES,
    can_cancel_plan,
    can_manage_billing,
    can_switch_plan,
    is_entitled,
    local_billing_enabled,
)
from app.db import get_db
from app.keys import (
    generate_api_key,
    is_active,
    key_status,
)
from app.models import ApiKey, UsageDaily, aware_utc, utcnow
from app.plans import SELF_SERVE_PLANS, limits_for
from app.templating import render_page

router = APIRouter()

_ERRORS = {
    "plan": "Choose an active plan before creating or using API keys.",
    "keylimit": "You reached the key limit for this plan. Revoke one first.",
    "notfound": "That API key no longer exists.",
    "choose-plan": "Choose Starter or Pro to continue.",
    "already": "You already have a subscription. Use Change plan or Manage billing.",
    "checkout": "Checkout could not be started. Try again or contact support.",
    "portal": "Stripe billing could not be opened. Try again or contact support.",
    "choose-other": "Choose a different plan to continue.",
    "change-plan": "Could not start the plan change. Try again or contact support.",
    "change-unavailable": "Plan switching is not available for this subscription.",
    "cancel-plan": "Could not open Stripe cancellation. Try Manage billing.",
    "cancel-unavailable": "Cancellation isn't available for this plan.",
}

_CHECKOUT_NOTICES = {
    "local": ("Plan activated locally. No payment was taken.", "ok"),
    "local-change": ("Plan changed locally. No payment was taken.", "ok"),
    "cancelled": ("Checkout was cancelled; no charge was made.", ""),
}


def _dashboard_flags(
    query: Mapping[str, str], context: dict[str, Any]
) -> dict[str, bool]:
    """Derive the one-shot dashboard view flags from the query string."""
    return {
        "awaiting_checkout": (
            query.get("checkout") == "success" and not context["entitled"]
        ),
        "confirming_preview_cancellation": (
            context["can_cancel_plan"]
            and context["local_billing"]
            and query.get("cancel") == "1"
        ),
        "comparing_plans": (context["can_switch_plan"] and query.get("plans") == "1"),
    }


def _dashboard_notice(
    query: Mapping[str, str], context: dict[str, Any]
) -> tuple[str, str]:
    """Return the dashboard notice and tone for a checkout or billing result."""
    checkout = query.get("checkout")
    if checkout == "success":
        if context["entitled"]:
            return "Your plan is active.", "ok"
        return "Stripe is confirming your subscription. Refresh in a moment.", "warning"
    billing = query.get("billing")
    if (
        billing == "cancellation"
        and context["can_cancel_plan"]
        and not context["local_billing"]
    ):
        return "Stripe is syncing your cancellation. Refresh in a moment.", "warning"
    if (
        billing == "preview-canceled"
        and context["local_billing"]
        and context["org"].subscription_status == "canceled"
    ):
        return "Preview plan canceled. Your account and keys were kept.", "ok"
    return _CHECKOUT_NOTICES.get(checkout or "", ("", ""))


def _usage_series(db: Session, org_id: uuid.UUID, today: date) -> list[int]:
    """Daily request counts for the current UTC month; missing days are 0."""
    month_start = today.replace(day=1)
    rows = db.execute(
        select(UsageDaily.day, func.sum(UsageDaily.requests))
        .where(UsageDaily.org_id == org_id, UsageDaily.day.between(month_start, today))
        .group_by(UsageDaily.day)
    )
    counts = {day: int(total or 0) for day, total in rows}
    span = (today - month_start).days + 1
    return [
        counts.get(month_start + timedelta(days=offset), 0) for offset in range(span)
    ]


def _dashboard_context(
    request: Request, db: Session, ctx: auth.SessionContext, **extra: Any
) -> dict[str, Any]:
    now = utcnow()
    keys = list(
        db.scalars(
            select(ApiKey)
            .where(ApiKey.org_id == ctx.org.id)
            .order_by(ApiKey.created_at.desc())
        )
    )
    org = ctx.org
    settings = request.app.state.settings
    gateway = request.app.state.billing
    limits = limits_for(org.plan)
    cancel_at = aware_utc(org.cancel_at)
    usage_series = _usage_series(db, org.id, now.date())
    used = sum(usage_series)
    quota = limits.request_quota if limits else 0
    percent = min(100, round(used * 100 / quota)) if quota else 0
    rows = [
        {
            "key": key,
            "status": key_status(key, now),
            "active": is_active(key, now),
            "last_used": aware_utc(key.last_used_at),
        }
        for key in keys
    ]
    return {
        "org": org,
        "entitled": is_entitled(org, now),
        "subscription_active": org.subscription_status in ACTIVE_SUBSCRIPTION_STATUSES,
        "limits": limits,
        "plans": SELF_SERVE_PLANS,
        "can_checkout": org.subscription_status not in CHECKOUT_BLOCKING_STATUSES,
        "can_switch_plan": can_switch_plan(org, gateway, settings),
        "can_manage_billing": can_manage_billing(org, gateway),
        "can_cancel_plan": can_cancel_plan(org, gateway, settings),
        "local_billing": local_billing_enabled(org, gateway, settings),
        "cancel_at": cancel_at,
        "cancellation_ended": cancel_at is not None and cancel_at <= now,
        "used": used,
        "quota": quota,
        "percent": percent,
        "usage_series": usage_series,
        "keys": rows,
        "active_keys": sum(1 for row in rows if row["active"]),
        "period_end": aware_utc(org.current_period_end),
        **extra,
    }


@router.get("/dashboard", response_class=HTMLResponse)
def dashboard_page(
    request: Request, db: Annotated[Session, Depends(get_db)]
) -> HTMLResponse:
    ctx = auth.current_session(request, db)
    query = request.query_params
    context = _dashboard_context(
        request, db, ctx, error_message=_ERRORS.get(query.get("error") or "")
    )
    context.update(_dashboard_flags(query, context))
    notice, tone = _dashboard_notice(query, context)
    context.update(notice=notice, notice_tone=tone)
    return render_page(
        request, "dashboard.html", "dashboard", signed_in=True, **context
    )


@router.post("/dashboard/keys", response_class=HTMLResponse)
def create_key(
    request: Request,
    db: Annotated[Session, Depends(get_db)],
    name: Annotated[str, Form()] = "",
    csrf_token: Annotated[str, Form()] = "",
) -> Response:
    ctx = auth.current_session(request, db)
    auth.check_csrf(request, csrf_token)
    if not is_entitled(ctx.org):
        return RedirectResponse("/dashboard?error=plan", status_code=303)
    limits = limits_for(ctx.org.plan)
    active = sum(
        1
        for key in db.scalars(select(ApiKey).where(ApiKey.org_id == ctx.org.id))
        if is_active(key, utcnow())
    )
    if limits and limits.max_keys is not None and active >= limits.max_keys:
        return RedirectResponse("/dashboard?error=keylimit", status_code=303)
    generated = generate_api_key()
    key = ApiKey(
        org_id=ctx.org.id,
        name=(name.strip()[:100] or "Default"),
        prefix=generated.prefix,
        secret_hash=generated.secret_hash,
        created_by=ctx.user.id,
    )
    db.add(key)
    db.flush()
    audit.record(
        db,
        org_id=ctx.org.id,
        actor_user_id=ctx.user.id,
        action="key.created",
        target_type="api_key",
        target_id=key.id,
        metadata={"name": key.name},
    )
    db.commit()
    return render_page(
        request,
        "key_created.html",
        "dashboard",
        signed_in=True,
        **_dashboard_context(
            request, db, ctx, plaintext=generated.full, created_name=key.name
        ),
    )


@router.post("/dashboard/keys/{key_id}/revoke")
def revoke_key(
    key_id: uuid.UUID,
    request: Request,
    db: Annotated[Session, Depends(get_db)],
    csrf_token: Annotated[str, Form()] = "",
) -> Response:
    ctx = auth.current_session(request, db)
    auth.check_csrf(request, csrf_token)
    key = db.get(ApiKey, key_id)
    if key is None or key.org_id != ctx.org.id:
        return RedirectResponse("/dashboard?error=notfound", status_code=303)
    if key.revoked_at is None:
        key.revoked_at = utcnow()
        audit.record(
            db,
            org_id=ctx.org.id,
            actor_user_id=ctx.user.id,
            action="key.revoked",
            target_type="api_key",
            target_id=key.id,
            metadata={"prefix": key.prefix},
        )
        db.commit()
        request.app.state.key_cache.invalidate_key(str(ctx.org.id), key.prefix)
    return RedirectResponse("/dashboard", status_code=303)


@router.post("/dashboard/keys/{key_id}/rotate", response_class=HTMLResponse)
def rotate_key(
    key_id: uuid.UUID,
    request: Request,
    db: Annotated[Session, Depends(get_db)],
    csrf_token: Annotated[str, Form()] = "",
) -> Response:
    ctx = auth.current_session(request, db)
    auth.check_csrf(request, csrf_token)
    if not is_entitled(ctx.org):
        return RedirectResponse("/dashboard?error=plan", status_code=303)
    old = db.get(ApiKey, key_id)
    if old is None or old.org_id != ctx.org.id:
        return RedirectResponse("/dashboard?error=notfound", status_code=303)
    limits = limits_for(ctx.org.plan)
    if limits and limits.max_keys is not None:
        active = sum(
            1
            for key in db.scalars(select(ApiKey).where(ApiKey.org_id == ctx.org.id))
            if key.id != old.id and is_active(key, utcnow())
        )
        if active >= limits.max_keys:
            return RedirectResponse("/dashboard?error=keylimit", status_code=303)
    generated = generate_api_key()
    old.expires_at = utcnow() + timedelta(hours=24)
    new = ApiKey(
        org_id=ctx.org.id,
        name=old.name,
        prefix=generated.prefix,
        secret_hash=generated.secret_hash,
        created_by=ctx.user.id,
    )
    db.add(new)
    db.flush()
    audit.record(
        db,
        org_id=ctx.org.id,
        actor_user_id=ctx.user.id,
        action="key.rotated",
        target_type="api_key",
        target_id=new.id,
        metadata={"old_prefix": old.prefix, "new_prefix": new.prefix},
    )
    db.commit()
    request.app.state.key_cache.invalidate_key(str(ctx.org.id), old.prefix)
    return render_page(
        request,
        "key_created.html",
        "dashboard",
        signed_in=True,
        **_dashboard_context(
            request,
            db,
            ctx,
            plaintext=generated.full,
            created_name=new.name,
            rotated=True,
        ),
    )
