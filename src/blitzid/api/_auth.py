"""Hosted API-key authentication, rate limits, and monthly quotas.

Self-hosted deployments leave ``BLITZID_API_AUTH_REQUIRED=0`` (the
default); the dependency then does nothing and no Postgres is needed.
Hosted deployments set it to 1, provide ``BLITZID_SAAS_DATABASE_URL``,
and the resolver fails closed when the key store or Redis is down.

``PLAN_LIMITS`` is the single enforcement source for quotas and rate
limits. The web dashboard mirrors the same numbers in
``web/app/plans.py``; a contract test in ``tests/test_api_auth.py``
fails when the two drift.
"""

from __future__ import annotations

import contextlib
import hashlib
import hmac
import importlib
import json
import logging
from datetime import UTC, datetime, timedelta
from typing import Annotated, Any, cast

import redis
from fastapi import BackgroundTasks, Header, Request

from blitzid.exceptions import BlitzIDError

_LOG = logging.getLogger(__name__)

_KEY_PREFIX = "blz_live_"

#: Unknown-key cache TTL and the global cap on cache-miss database lookups.
_MISS_CACHE_SECONDS = 30
_LOOKUP_LIMIT_PER_MINUTE = 1200

#: Monthly request quota and per-minute rate limit per plan (``None`` =
#: unlimited). Enforced here; mirrored for display by the web dashboard.
PLAN_LIMITS: dict[str, tuple[int | None, int | None]] = {
    "starter": (1_000, 30),
    "pro": (10_000, 120),
    "enterprise": (None, None),
}


class SaasError(Exception):
    """Authentication/enforcement failure with an OpenAPI error body."""

    def __init__(
        self,
        status_code: int,
        code: str,
        message: str,
        *,
        retry_after: int | None = None,
    ) -> None:
        super().__init__(message)
        self.status_code = status_code
        self.code = code
        self.message = message
        self.retry_after = retry_after


def _invalid() -> SaasError:
    return SaasError(401, "invalid_api_key", "Invalid API key.")


def parse_authorization(value: str | None) -> tuple[str, str]:
    """Split ``Bearer blz_live_<prefix>.<secret>`` into its two halves."""
    if not value or not value.lower().startswith("bearer "):
        raise _invalid()
    key = value[7:].strip()
    if not key.startswith(_KEY_PREFIX):
        raise _invalid()
    rest = key[len(_KEY_PREFIX) :]
    prefix, sep, secret = rest.partition(".")
    if not sep or len(prefix) != 8 or len(secret) < 20:
        raise _invalid()
    return prefix, secret


def _hash_secret(secret: str) -> str:
    return hashlib.sha256(secret.encode("utf-8")).hexdigest()


def _as_utc(value: Any) -> datetime | None:
    if value is None:
        return None
    if isinstance(value, str):
        try:
            value = datetime.fromisoformat(value)
        except ValueError:
            return None
    if not isinstance(value, datetime):
        return None
    return value if value.tzinfo is not None else value.replace(tzinfo=UTC)


def _entitled(record: dict[str, Any], now: datetime) -> bool:
    if record.get("plan") not in PLAN_LIMITS:
        return False
    cancel_at = _as_utc(record.get("cancel_at"))
    if cancel_at is not None and cancel_at <= now:
        return False
    status = record.get("subscription_status")
    if status in {"active", "trialing"}:
        return True
    grace = _as_utc(record.get("grace_until"))
    return status == "past_due" and grace is not None and grace > now


class PostgresApiAuth:
    """Postgres lookup + Redis cache + rate/quota enforcement."""

    def __init__(self, database_url: str, redis_url: str) -> None:
        psycopg_rows = importlib.import_module("psycopg.rows")
        psycopg_pool = importlib.import_module("psycopg_pool")
        self._dict_row = psycopg_rows.dict_row
        self._pool = psycopg_pool.ConnectionPool(
            database_url, min_size=1, max_size=4, timeout=2, open=False
        )
        self._pool.open(wait=False)
        self._redis = redis.Redis.from_url(
            redis_url,
            decode_responses=True,
            socket_timeout=2,
            socket_connect_timeout=2,
        )

    def close(self) -> None:
        self._pool.close()
        self._redis.close()

    def ready(self) -> bool:
        try:
            with self._pool.connection() as conn:
                conn.execute("SELECT cancel_at FROM organizations LIMIT 0")
            self._redis.ping()
        except Exception:
            return False
        return True

    def authenticate_and_enforce(
        self, authorization: str | None
    ) -> tuple[dict[str, Any], datetime] | None:
        """Authenticate a key and enforce plan, rate, and quota limits.

        Returns:
            The cached key record and request time when the key is valid,
            so the caller can record usage after the response.

        Raises:
            SaasError: ``401`` invalid key, ``403`` no active plan,
                ``429`` rate or quota exceeded, ``503`` enforcement
                backend unavailable (fails closed).
        """
        prefix, secret = parse_authorization(authorization)
        now = datetime.now(UTC)
        record = self._cache_get(prefix)
        if record is None:
            if self._cache_is_missing(prefix):
                raise _invalid()
            self._allow_lookup(now)
            record = self._lookup(prefix)
            if record is None:
                self._cache_mark_missing(prefix)
                raise _invalid()
            self._cache_set(prefix, record)
        if record is None:
            raise _invalid()
        expected = str(record.get("secret_hash", ""))
        if not hmac.compare_digest(_hash_secret(secret), expected):
            raise _invalid()
        if record.get("revoked_at") is not None:
            raise _invalid()
        expires = _as_utc(record.get("expires_at"))
        if expires is not None and expires <= now:
            raise _invalid()
        if not _entitled(record, now):
            raise SaasError(403, "plan_required", "An active plan is required.")
        quota, per_minute = PLAN_LIMITS[str(record["plan"])]
        self._enforce_limits(record, quota, per_minute, now)
        return record, now

    def _lookup(self, prefix: str) -> dict[str, Any] | None:
        try:
            with (
                self._pool.connection() as conn,
                conn.cursor(row_factory=self._dict_row) as cur,
            ):
                cur.execute(
                    """
                        SELECT k.id::text AS key_id,
                               k.org_id::text AS org_id,
                               k.prefix,
                               k.secret_hash,
                               k.revoked_at,
                               k.expires_at,
                               o.plan,
                               o.subscription_status,
                               o.current_period_end,
                               o.cancel_at,
                               o.grace_until
                        FROM api_keys k
                        JOIN organizations o ON o.id = k.org_id
                        WHERE k.prefix = %s
                        """,
                    (prefix,),
                )
                row = cur.fetchone()
        except Exception as exc:
            _LOG.error("API-key lookup failed: %s", exc)
            raise SaasError(
                503,
                "auth_unavailable",
                "Authentication is temporarily unavailable.",
            ) from exc
        return cast(dict[str, Any] | None, row)

    def _cache_get(self, prefix: str) -> dict[str, Any] | None:
        try:
            raw = self._redis.get(f"apikey:{prefix}")
        except redis.RedisError:
            return None
        if not raw:
            return None
        try:
            return cast(dict[str, Any], json.loads(raw))
        except ValueError:
            return None

    def _cache_set(self, prefix: str, record: dict[str, Any]) -> None:
        payload: dict[str, Any] = {}
        for key, value in record.items():
            if isinstance(value, datetime):
                payload[key] = value.isoformat()
            else:
                payload[key] = value
        org_id = str(record.get("org_id", ""))
        try:
            self._redis.delete(f"apikey_miss:{prefix}")
            self._redis.set(f"apikey:{prefix}", json.dumps(payload), ex=60)
            if org_id:
                self._redis.sadd(f"orgkeys:{org_id}", prefix)
                self._redis.expire(f"orgkeys:{org_id}", 8 * 24 * 3600)
        except redis.RedisError as exc:
            _LOG.warning("API-key cache write failed: %s", exc)

    def _cache_is_missing(self, prefix: str) -> bool:
        try:
            return bool(self._redis.exists(f"apikey_miss:{prefix}"))
        except redis.RedisError:
            return False

    def _cache_mark_missing(self, prefix: str) -> None:
        with contextlib.suppress(redis.RedisError):
            self._redis.set(f"apikey_miss:{prefix}", "1", ex=_MISS_CACHE_SECONDS)

    def _allow_lookup(self, now: datetime) -> None:
        key = f"authlookup:{int(now.timestamp()) // 60}"
        try:
            count = self._redis.incr(key)
            if count == 1:
                self._redis.expire(key, 120)
        except redis.RedisError as exc:
            raise SaasError(
                503,
                "auth_unavailable",
                "Authentication is temporarily unavailable.",
            ) from exc
        if count > _LOOKUP_LIMIT_PER_MINUTE:
            raise SaasError(
                429,
                "rate_limit_exceeded",
                "Too many authentication attempts.",
                retry_after=60,
            )

    def _enforce_limits(
        self,
        record: dict[str, Any],
        quota: int | None,
        per_minute: int | None,
        now: datetime,
    ) -> None:
        key_id = str(record["key_id"])
        org_id = str(record["org_id"])
        try:
            if per_minute is not None:
                minute = int(now.timestamp()) // 60
                rate_key = f"rl:{key_id}:{minute}"
                count = self._redis.incr(rate_key)
                if count == 1:
                    self._redis.expire(rate_key, 120)
                if count > per_minute:
                    raise SaasError(
                        429,
                        "rate_limited",
                        "Too many requests.",
                        retry_after=60 - (int(now.timestamp()) % 60),
                    )
            if quota is not None:
                month = now.strftime("%Y%m")
                quota_key = f"quota:{org_id}:{month}"
                used = self._redis.incr(quota_key)
                if used == 1:
                    self._redis.expire(quota_key, 35 * 24 * 3600)
                if used > quota:
                    next_month = (now.replace(day=1) + timedelta(days=32)).replace(
                        day=1
                    )
                    raise SaasError(
                        429,
                        "quota_exceeded",
                        "Monthly request limit reached.",
                        retry_after=max(1, int((next_month - now).total_seconds())),
                    )
        except redis.RedisError as exc:
            _LOG.error("rate limiting unavailable: %s", exc)
            raise SaasError(
                503,
                "rate_limit_unavailable",
                "Rate limiting is temporarily unavailable.",
            ) from exc

    def record_usage(self, record: dict[str, Any], now: datetime) -> None:
        try:
            with self._pool.connection() as conn, conn.cursor() as cur:
                cur.execute(
                    """
                        INSERT INTO usage_daily
                            (org_id, api_key_id, day, requests)
                        VALUES (%s::uuid, %s::uuid, %s, 1)
                        ON CONFLICT (org_id, api_key_id, day)
                        DO UPDATE SET requests = usage_daily.requests + 1
                        """,
                    (record["org_id"], record["key_id"], now.date()),
                )
                cur.execute(
                    """
                        UPDATE api_keys
                        SET last_used_at = %s
                        WHERE id = %s::uuid
                          AND (
                            last_used_at IS NULL
                            OR last_used_at < %s - interval '1 minute'
                          )
                        """,
                    (now, record["key_id"], now),
                )
        except Exception as exc:
            _LOG.warning("usage recording failed: %s", exc)


def build_api_auth(database_url: str, redis_url: str) -> PostgresApiAuth:
    """Build the resolver.

    Raises:
        BlitzIDError: If the key store URL is missing; hosted auth
            cannot work without it.
    """
    if not database_url:
        raise BlitzIDError(
            "BLITZID_SAAS_DATABASE_URL is required when API auth is enabled"
        )
    return PostgresApiAuth(database_url, redis_url)


def require_api_key(
    request: Request,
    background: BackgroundTasks,
    authorization: Annotated[str | None, Header()] = None,
) -> None:
    """FastAPI dependency for protected routes (no-op when auth is off)."""
    config = getattr(request.app.state, "config", None)
    if config is None or not config.auth_required:
        return
    resolver = getattr(request.app.state, "api_auth", None)
    if resolver is None:
        raise SaasError(
            503, "auth_unavailable", "Authentication is temporarily unavailable."
        )
    usage = resolver.authenticate_and_enforce(authorization)
    if usage is not None:
        background.add_task(resolver.record_usage, *usage)
