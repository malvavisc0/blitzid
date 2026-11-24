"""OAuth sign-in, callback, and logout routes."""

from __future__ import annotations

import logging
from typing import Annotated, cast

from fastapi import APIRouter, Depends, Form, HTTPException, Request
from fastapi.responses import RedirectResponse, Response
from sqlalchemy.orm import Session

from app import auth
from app.db import get_db

_LOG = logging.getLogger(__name__)

router = APIRouter()


@router.get("/auth/login/{provider}")
async def start_oauth(provider: str, request: Request) -> Response:
    """Redirect to the provider's consent screen."""
    configured = request.app.state.oauth_providers
    selected = configured.get(provider)
    if selected is None:
        raise HTTPException(status_code=404, detail="Unknown sign-in provider")
    try:
        return cast(Response, await selected.authorize_redirect(request))
    except Exception:
        _LOG.exception("OAuth start failed for %s", provider)
        return RedirectResponse("/login?error=oauth", status_code=303)


@router.get("/auth/callback/{provider}")
async def oauth_callback(
    provider: str, request: Request, db: Annotated[Session, Depends(get_db)]
) -> Response:
    """Exchange the code, provision the account, and start a session."""
    configured = request.app.state.oauth_providers
    selected = configured.get(provider)
    if selected is None:
        raise HTTPException(status_code=404, detail="Unknown sign-in provider")
    try:
        profile = await selected.fetch_profile(request)
    except Exception:
        _LOG.exception("OAuth callback failed for %s", provider)
        return RedirectResponse("/login?error=oauth", status_code=303)
    try:
        user, org = auth.provision_user(db, profile)
        db.commit()
    except auth.AuthError:
        db.rollback()
        return RedirectResponse("/login?error=email", status_code=303)
    except Exception:
        db.rollback()
        _LOG.exception("Account provisioning failed")
        return RedirectResponse("/login?error=oauth", status_code=303)
    auth.sign_in_session(request, user, org)
    return RedirectResponse("/dashboard", status_code=303)


@router.post("/auth/logout")
def logout(
    request: Request, csrf_token: Annotated[str, Form()] = ""
) -> RedirectResponse:
    """Clear the session."""
    auth.check_csrf(request, csrf_token)
    auth.sign_out_session(request)
    return RedirectResponse("/", status_code=303)
