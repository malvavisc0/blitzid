"""Public pages: landing, pricing, login, legal, and health."""

from __future__ import annotations

from fastapi import APIRouter, Request
from fastapi.responses import HTMLResponse, RedirectResponse, Response

from app import auth
from app.plans import SELF_SERVE_PLANS
from app.templating import render_page

router = APIRouter()


@router.get("/", response_class=HTMLResponse)
def landing(request: Request) -> HTMLResponse:
    """The one-page marketing site."""
    return render_page(request, "landing.html", "home", plans=SELF_SERVE_PLANS)


@router.get("/pricing")
def pricing() -> RedirectResponse:
    """Pricing lives in the landing page for now."""
    return RedirectResponse("/#pricing", status_code=307)


@router.get("/login", response_class=HTMLResponse)
def login(request: Request) -> Response:
    """Sign-in choices; signed-in users go straight to the dashboard."""
    if auth.is_signed_in(request):
        return RedirectResponse("/dashboard", status_code=303)
    return render_page(
        request,
        "login.html",
        "login",
        signed_in=False,
        providers=request.app.state.oauth_providers,
        error=request.query_params.get("error"),
    )


@router.get("/privacy", response_class=HTMLResponse)
def privacy(request: Request) -> HTMLResponse:
    return render_page(request, "legal.html", "privacy", title="Privacy")


@router.get("/terms", response_class=HTMLResponse)
def terms(request: Request) -> HTMLResponse:
    return render_page(request, "legal.html", "terms", title="Terms")


@router.get("/healthz")
def healthz(request: Request) -> dict[str, str]:
    """Liveness for the container and reverse proxy."""
    settings = request.app.state.settings
    return {
        "status": "ok",
        "service": "web",
        "version": settings.version,
        "environment": settings.environment,
    }
