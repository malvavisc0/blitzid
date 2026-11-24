"""API-key generation, hashing, and status helpers."""

from __future__ import annotations

import hashlib
import secrets
from dataclasses import dataclass
from datetime import datetime

from app.models import ApiKey, aware_utc

KEY_PREFIX = "blz_live_"


@dataclass(frozen=True)
class GeneratedKey:
    """A freshly minted key: plaintext is shown once and never stored."""

    full: str
    prefix: str
    secret_hash: str


def generate_api_key() -> GeneratedKey:
    """Mint ``blz_live_<8-char prefix>.<43-char secret>``."""
    prefix = secrets.token_urlsafe(6)[:8]
    secret = secrets.token_urlsafe(32)
    return GeneratedKey(
        full=f"{KEY_PREFIX}{prefix}.{secret}",
        prefix=prefix,
        secret_hash=hash_secret(secret),
    )


def hash_secret(secret: str) -> str:
    """SHA-256 hex digest of the secret half."""
    return hashlib.sha256(secret.encode("utf-8")).hexdigest()


def is_active(key: ApiKey, now: datetime) -> bool:
    """True when the key is neither revoked nor past its rotation expiry."""
    if key.revoked_at is not None:
        return False
    expires = aware_utc(key.expires_at)
    return expires is None or expires > now


def key_status(key: ApiKey, now: datetime) -> str:
    """Human-readable status for the dashboard table."""
    if key.revoked_at is not None:
        return "Revoked"
    expires = aware_utc(key.expires_at)
    if expires is not None and expires <= now:
        return "Expired"
    if expires is not None:
        return "Rotating"
    return "Active"
