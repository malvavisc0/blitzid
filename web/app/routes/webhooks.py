"""Stripe webhook: signature verification, idempotency, entitlements."""

from __future__ import annotations

import logging
from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, Request
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session
from starlette.concurrency import run_in_threadpool

from app.billing import (
    BillingEvent,
    BillingGateway,
    BillingUnavailable,
    apply_event,
)
from app.cache import KeyCache
from app.config import Settings
from app.db import get_db
from app.models import StripeEvent, utcnow

_LOG = logging.getLogger(__name__)

router = APIRouter()


def _process_webhook(
    db: Session,
    event: BillingEvent,
    gateway: BillingGateway,
    settings: Settings,
    key_cache: KeyCache,
) -> None:
    if db.get(StripeEvent, event.id) is not None:
        return
    record = StripeEvent(id=event.id, type=event.type)
    db.add(record)
    try:
        org = apply_event(db, event, gateway, settings)
        record.processed_at = utcnow()
        db.commit()
    except IntegrityError:
        db.rollback()
        return
    except BillingUnavailable as exc:
        db.rollback()
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except Exception as exc:
        db.rollback()
        _LOG.exception("Stripe webhook failed")
        raise HTTPException(
            status_code=500, detail="Webhook processing failed"
        ) from exc
    if org is not None:
        key_cache.invalidate_org(str(org.id))


@router.post("/webhooks/stripe")
async def stripe_webhook(
    request: Request, db: Annotated[Session, Depends(get_db)]
) -> dict[str, bool]:
    """Verify, de-duplicate, and apply one Stripe event."""
    payload = await request.body()
    signature = request.headers.get("stripe-signature", "")
    gateway = request.app.state.billing
    try:
        event = gateway.parse_event(payload, signature)
    except BillingUnavailable as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except Exception as exc:
        _LOG.warning("Stripe webhook rejected: %s", exc)
        raise HTTPException(status_code=400, detail="Invalid Stripe event") from exc
    await run_in_threadpool(
        _process_webhook,
        db,
        event,
        gateway,
        request.app.state.settings,
        request.app.state.key_cache,
    )
    return {"received": True}
