"""FastAPI entrypoint for the BlitzID website and hosted dashboard."""

from __future__ import annotations

import secrets
from collections.abc import Awaitable, Callable

import redis
from fastapi import FastAPI, Request, Response
from fastapi.responses import HTMLResponse, PlainTextResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from starlette.middleware.sessions import SessionMiddleware

from app import auth
from app.billing import BillingGateway, BillingUnavailable, NullBilling, StripeBilling
from app.cache import KeyCache
from app.config import STATIC_DIR, Settings
from app.db import Database
from app.routes import auth as auth_routes
from app.routes import billing as billing_routes
from app.routes import dashboard, docs, pages, webhooks
from app.templating import render_page


def create_app(settings: Settings | None = None) -> FastAPI:
    """Build the site with optional settings (tests inject their own)."""
    resolved = settings or Settings.from_env()
    if resolved.environment == "production" and not resolved.session_secret:
        raise RuntimeError("SESSION_SECRET must be set in production")
    app = FastAPI(
        title="BlitzID website",
        docs_url=None,
        redoc_url=None,
        openapi_url=None,
    )
    app.state.settings = resolved

    database = Database(resolved.database_url)
    if (
        resolved.database_url.startswith("sqlite")
        and resolved.environment != "production"
    ):
        database.create_all()
    app.state.db = database

    app.state.oauth_providers = auth.build_providers(resolved)
    billing: BillingGateway
    try:
        billing = (
            StripeBilling(resolved) if resolved.stripe_secret_key else NullBilling()
        )
    except BillingUnavailable:
        billing = NullBilling()
    app.state.billing = billing

    try:
        redis_client: redis.Redis | None = redis.Redis.from_url(
            resolved.redis_url,
            decode_responses=True,
            socket_timeout=2,
            socket_connect_timeout=2,
        )
    except (ValueError, redis.RedisError):
        redis_client = None
    app.state.key_cache = KeyCache(redis_client)

    app.add_middleware(
        SessionMiddleware,
        # Development without SESSION_SECRET signs with a per-process
        # random key: sessions vanish on restart, which is fine locally.
        secret_key=resolved.session_secret or secrets.token_urlsafe(32),
        session_cookie="blitzid_session",
        max_age=7 * 24 * 60 * 60,
        same_site="lax",
        https_only=resolved.secure_cookies,
    )
    app.include_router(pages.router)
    app.include_router(docs.router)
    app.include_router(auth_routes.router)
    app.include_router(dashboard.router)
    app.include_router(billing_routes.router)
    app.include_router(webhooks.router)

    @app.middleware("http")
    async def _security_headers(
        request: Request, call_next: Callable[[Request], Awaitable[Response]]
    ) -> Response:
        response = await call_next(request)
        response.headers.setdefault("X-Content-Type-Options", "nosniff")
        response.headers.setdefault(
            "Referrer-Policy", "strict-origin-when-cross-origin"
        )
        response.headers.setdefault("X-Frame-Options", "DENY")
        if request.url.path.startswith(("/login", "/dashboard", "/billing", "/auth")):
            response.headers["Cache-Control"] = "no-store"
        return response

    @app.exception_handler(auth.NotAuthenticated)
    async def _not_authenticated(
        request: Request, exc: auth.NotAuthenticated
    ) -> RedirectResponse:
        return RedirectResponse("/login", status_code=303)

    @app.get("/favicon.ico", include_in_schema=False)
    def favicon() -> RedirectResponse:
        return RedirectResponse("/static/favicon.svg", status_code=308)

    @app.get("/robots.txt", response_class=PlainTextResponse, include_in_schema=False)
    def robots() -> str:
        return "User-agent: *\nDisallow: /dashboard\nDisallow: /billing\n"

    @app.exception_handler(404)
    def _not_found(request: Request, exc: Exception) -> HTMLResponse:
        return render_page(
            request,
            "error.html",
            "404",
            status=404,
            message="Page not found",
            status_code=404,
        )

    app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")
    return app


app = create_app()
