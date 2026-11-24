"""Render pages with the shared, session-aware site shell."""

from __future__ import annotations

from typing import Any

from fastapi import Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates

from app import auth
from app.config import TEMPLATES_DIR

_templates = Jinja2Templates(directory=str(TEMPLATES_DIR))


def render_page(
    request: Request,
    template: str,
    page: str,
    *,
    signed_in: bool | None = None,
    status_code: int = 200,
    **context: Any,
) -> HTMLResponse:
    # Protected routes already checked membership and can skip a second lookup.
    if signed_in is None:
        signed_in = auth.is_signed_in(request)
    return _templates.TemplateResponse(
        request,
        template,
        {
            **context,
            "page": page,
            "signed_in": signed_in,
            "csrf_token": auth.csrf_token(request) if signed_in else "",
        },
        status_code=status_code,
        headers={"Cache-Control": "no-store"},
    )
