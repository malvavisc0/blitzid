"""Redis cache invalidation shared with the API key resolver."""

from __future__ import annotations

import logging

import redis

_LOG = logging.getLogger(__name__)


class KeyCache:
    """Delete cached API-key entitlements when keys or plans change."""

    def __init__(self, client: redis.Redis | None) -> None:
        self._client = client

    def invalidate_key(self, org_id: str, prefix: str) -> None:
        if self._client is None:
            return
        try:
            self._client.delete(f"apikey:{prefix}")
            self._client.srem(f"orgkeys:{org_id}", prefix)
        except redis.RedisError as exc:
            _LOG.warning("key cache invalidation failed: %s", exc)

    def invalidate_org(self, org_id: str) -> None:
        if self._client is None:
            return
        try:
            raw = self._client.smembers(f"orgkeys:{org_id}")
            prefixes = {p.decode() if isinstance(p, bytes) else str(p) for p in raw}
            if prefixes:
                self._client.delete(*(f"apikey:{p}" for p in prefixes))
            self._client.delete(f"orgkeys:{org_id}")
        except redis.RedisError as exc:
            _LOG.warning("org cache invalidation failed: %s", exc)
