"""OAuth sign-in, sessions, CSRF, and organization provisioning."""

from __future__ import annotations

import logging
import secrets
import uuid
from dataclasses import dataclass
from typing import Any, Protocol, cast

from authlib.integrations.starlette_client import OAuth  # type: ignore[import-untyped]
from fastapi import HTTPException, Request, Response
from sqlalchemy import func, select
from sqlalchemy.orm import Session

from app.config import Settings
from app.db import Database
from app.models import Identity, Membership, Organization, User

_LOG = logging.getLogger(__name__)

_SESSION_USER = "user_id"
_SESSION_ORG = "org_id"
_SESSION_CSRF = "csrf_token"

CSRF_ERROR = "Invalid or missing CSRF token."


class AuthError(RuntimeError):
    """Sign-in could not be completed or the provider data is unusable."""


class NotAuthenticated(Exception):
    """No valid session; handled as a redirect to /login."""


@dataclass(frozen=True)
class OAuthProfile:
    provider: str
    subject: str
    email: str
    email_verified: bool
    name: str | None
    avatar_url: str | None


@dataclass(frozen=True)
class SessionContext:
    user: User
    org: Organization


class OAuthProvider(Protocol):
    name: str

    async def authorize_redirect(self, request: Request) -> Response: ...

    async def fetch_profile(self, request: Request) -> OAuthProfile: ...


class AuthlibProvider:
    """Wrap one Authlib Starlette client as a provider adapter."""

    def __init__(self, name: str, client: Any, settings: Settings) -> None:
        self.name = name
        self._client = client
        self._settings = settings

    async def authorize_redirect(self, request: Request) -> Response:
        redirect_uri = f"{self._settings.app_base_url}/auth/callback/{self.name}"
        return cast(
            Response, await self._client.authorize_redirect(request, redirect_uri)
        )

    async def fetch_profile(self, request: Request) -> OAuthProfile:
        token = await self._client.authorize_access_token(request)
        if self.name == "google":
            return self._google_profile(token)
        return await self._github_profile(token)

    def _google_profile(self, token: dict[str, Any]) -> OAuthProfile:
        info = token.get("userinfo") or {}
        subject = info.get("sub")
        email = info.get("email")
        if not subject or not email:
            raise AuthError("Google did not return an email address")
        if not info.get("email_verified"):
            raise AuthError("Google email is not verified")
        return OAuthProfile(
            provider="google",
            subject=str(subject),
            email=str(email).lower(),
            email_verified=True,
            name=info.get("name"),
            avatar_url=info.get("picture"),
        )

    async def _github_profile(self, token: dict[str, Any]) -> OAuthProfile:
        emails = await self._github_json("user/emails", token, "email addresses")
        primary = _primary_github_email(emails)
        profile = await self._github_json("user", token, "a profile")
        subject = profile.get("id")
        if not subject:
            raise AuthError("GitHub did not return a user id")
        return OAuthProfile(
            provider="github",
            subject=str(subject),
            email=str(primary["email"]).lower(),
            email_verified=True,
            name=profile.get("name") or profile.get("login"),
            avatar_url=profile.get("avatar_url"),
        )

    async def _github_json(self, path: str, token: dict[str, Any], what: str) -> Any:
        """Fetch one GitHub API object, normalizing failures to AuthError.

        Returns the decoded JSON body (GitHub's schema, third-party
        boundary).
        """
        response = await self._client.get(path, token=token)
        if response.status_code != 200:
            raise AuthError(f"GitHub did not return {what}")
        return response.json()


def _primary_github_email(emails: list[dict[str, Any]]) -> dict[str, Any]:
    """Pick the verified primary email, falling back to any verified one."""
    verified = [e for e in emails if e.get("verified")]
    primary = next(
        (e for e in verified if e.get("primary")),
        verified[0] if verified else None,
    )
    if primary is None:
        raise AuthError("GitHub account has no verified email")
    return primary


def build_providers(settings: Settings) -> dict[str, OAuthProvider]:
    """Register the providers that have credentials configured."""
    providers: dict[str, OAuthProvider] = {}
    oauth = OAuth()
    if settings.google_client_id and settings.google_client_secret:
        oauth.register(
            name="google",
            client_id=settings.google_client_id,
            client_secret=settings.google_client_secret,
            server_metadata_url=(
                "https://accounts.google.com/.well-known/openid-configuration"
            ),
            client_kwargs={"scope": "openid email profile"},
        )
        providers["google"] = AuthlibProvider("google", oauth.google, settings)
    if settings.github_client_id and settings.github_client_secret:
        oauth.register(
            name="github",
            client_id=settings.github_client_id,
            client_secret=settings.github_client_secret,
            authorize_url="https://github.com/login/oauth/authorize",
            access_token_url="https://github.com/login/oauth/access_token",
            api_base_url="https://api.github.com/",
            client_kwargs={"scope": "user:email"},
        )
        providers["github"] = AuthlibProvider("github", oauth.github, settings)
    return providers


def csrf_token(request: Request) -> str:
    """Return the session CSRF token, creating it on first use."""
    token = request.session.get(_SESSION_CSRF)
    if not token:
        token = secrets.token_urlsafe(32)
        request.session[_SESSION_CSRF] = token
    return str(token)


def check_csrf(request: Request, token: str | None) -> None:
    """Reject a state-changing request with a missing/mismatched token."""
    expected = request.session.get(_SESSION_CSRF)
    if (
        not expected
        or not token
        or not secrets.compare_digest(str(expected), str(token))
    ):
        raise HTTPException(status_code=403, detail=CSRF_ERROR)


def sign_in_session(request: Request, user: User, org: Organization) -> None:
    """Start a fresh session for a user and their personal organization."""
    request.session.clear()
    request.session[_SESSION_USER] = str(user.id)
    request.session[_SESSION_ORG] = str(org.id)
    request.session[_SESSION_CSRF] = secrets.token_urlsafe(32)


def sign_out_session(request: Request) -> None:
    request.session.clear()


def current_session(request: Request, db: Session) -> SessionContext:
    """Resolve the signed-in user/org, or raise NotAuthenticated."""
    raw_user = request.session.get(_SESSION_USER)
    raw_org = request.session.get(_SESSION_ORG)
    if not raw_user or not raw_org:
        raise NotAuthenticated
    try:
        user_id = uuid.UUID(str(raw_user))
        org_id = uuid.UUID(str(raw_org))
    except ValueError as exc:
        request.session.clear()
        raise NotAuthenticated from exc
    user = db.get(User, user_id)
    org = db.get(Organization, org_id)
    membership = db.get(Membership, (org_id, user_id))
    if user is None or org is None or membership is None:
        request.session.clear()
        raise NotAuthenticated
    return SessionContext(user=user, org=org)


def is_signed_in(request: Request) -> bool:
    """Check navigation state without opening a database session for visitors."""
    if not request.session.get(_SESSION_USER) or not request.session.get(_SESSION_ORG):
        return False
    database: Database = request.app.state.db
    with database.session_factory() as db:
        try:
            current_session(request, db)
        except NotAuthenticated:
            return False
    return True


def personal_org(db: Session, user: User) -> Organization:
    """Return the user's organization, creating a personal one if needed."""
    org = db.scalar(
        select(Organization)
        .join(Membership, Membership.org_id == Organization.id)
        .where(Membership.user_id == user.id)
        .order_by(Organization.created_at)
    )
    if org is not None:
        return org
    org = Organization(name=user.name or user.email.split("@")[0])
    db.add(org)
    db.flush()
    db.add(Membership(org_id=org.id, user_id=user.id, role="owner"))
    db.flush()
    return org


def provision_user(db: Session, profile: OAuthProfile) -> tuple[User, Organization]:
    """Sign in by provider identity, linking only on a verified email."""
    identity = db.scalar(
        select(Identity).where(
            Identity.provider == profile.provider,
            Identity.provider_subject == profile.subject,
        )
    )
    if identity is not None:
        user = db.get(User, identity.user_id)
        if user is not None:
            return user, personal_org(db, user)
        db.delete(identity)
        db.flush()

    if not profile.email_verified:
        raise AuthError("Provider email is not verified")
    user = db.scalar(
        select(User).where(func.lower(User.email) == profile.email.lower())
    )
    if user is None:
        user = User(
            email=profile.email.lower(),
            name=profile.name,
            avatar_url=profile.avatar_url,
        )
        db.add(user)
        db.flush()
    elif not user.name and profile.name:
        user.name = profile.name
    db.add(
        Identity(
            user_id=user.id,
            provider=profile.provider,
            provider_subject=profile.subject,
        )
    )
    db.flush()
    return user, personal_org(db, user)
