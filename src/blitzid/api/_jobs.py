"""Redis-backed job store for the blitzid API.

Jobs live entirely in Redis memory (the server runs with persistence
disabled): a queue list, a processing list, status/payload/result
string keys, a claim-lease key per running job, and a submission
registry zset that distinguishes expired/claimed jobs (``410``) from
unknown ids (``404``). Every string key carries an ``EXPIRE`` bounded
by the job TTL, and the submission registry is pruned to a grace
window, so memory stays bounded — nothing ever reaches disk.

Claim protocol: ``LMOVE`` pops a job id from the queue into the
processing list; results are written once and read via ``GETDEL``
(claim-once — every subsequent read sees the id as gone). A worker
crash is recovered by the lease sweeper: ids still in processing past
``BLITZID_API_JOB_LEASE_SECONDS`` are pushed back onto the queue.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from typing import Any

import redis

from ..exceptions import BlitzIDError

QUEUE_KEY = "blitzid:queue"
PROCESSING_KEY = "blitzid:processing"
STATUS_PREFIX = "blitzid:status:"
PAYLOAD_PREFIX = "blitzid:payload:"
RESULT_PREFIX = "blitzid:result:"
CLAIM_PREFIX = "blitzid:claim:"
SUBMITTED_KEY = "blitzid:submitted"


class QueueFullError(BlitzIDError):
    """The queue already holds the maximum number of jobs."""


@dataclass(frozen=True)
class JobSnapshot:
    """One job's state as seen by a status read.

    Attributes:
        status: ``"queued"``, ``"running"``, ``"done"`` (a result
            exists — claimed by this read), ``"gone"`` (expired or
            already claimed), or ``"missing"`` (unknown id).
        result: The stored result dict when status is ``"done"``.
    """

    status: str
    result: dict[str, Any] | None = None


def _text(value: Any) -> str:
    """Decode bytes-typed Redis replies into str."""
    return value.decode() if isinstance(value, bytes) else value


class RedisJobStore:
    """Async-job queue over Redis primitives.

    Args:
        client: A ``redis.Redis`` client (``decode_responses=False``).
        ttl_seconds: Lifetime of every job key (``EXPIRE``).
        max_queued: Queue capacity; :meth:`submit` raises
            :class:`QueueFullError` beyond it.
        lease_seconds: Claim lease; a running job whose lease expired
            is requeued by :meth:`sweep`.
    """

    def __init__(
        self,
        client: redis.Redis,
        ttl_seconds: int = 900,
        max_queued: int = 50,
        lease_seconds: int = 120,
    ) -> None:
        self._redis = client
        self._ttl = ttl_seconds
        self._max_queued = max_queued
        self._lease = lease_seconds

    @classmethod
    def from_url(
        cls, url: str, ttl_seconds: int, max_queued: int, lease_seconds: int
    ) -> RedisJobStore:
        """Build a store from a Redis URL (``redis://host:port/db``)."""
        client = redis.Redis.from_url(url, decode_responses=False)
        return cls(client, ttl_seconds, max_queued, lease_seconds)

    def ping(self) -> bool:
        """Whether Redis is reachable."""
        try:
            return bool(self._redis.ping())
        except redis.RedisError:
            return False

    def counts(self) -> dict[str, int]:
        """Queue, processing, and unclaimed-result counts for /health."""
        try:
            stored = sum(1 for _ in self._redis.scan_iter(match=RESULT_PREFIX + "*"))
            return {
                "queued": int(self._redis.llen(QUEUE_KEY)),
                "running": int(self._redis.llen(PROCESSING_KEY)),
                "stored": stored,
            }
        except redis.RedisError:
            return {"queued": 0, "running": 0, "stored": 0}

    def submit(self, job_id: str, payload: dict[str, Any]) -> None:
        """Store the payload and enqueue the job id.

        Raises:
            QueueFullError: If the queue is at capacity.
            redis.RedisError: If Redis is unreachable.
        """
        if int(self._redis.llen(QUEUE_KEY)) >= self._max_queued:
            raise QueueFullError(f"queue is full ({self._max_queued} queued jobs)")
        now = time.time()
        pipe = self._redis.pipeline(transaction=True)
        pipe.set(PAYLOAD_PREFIX + job_id, json.dumps(payload), ex=self._ttl)
        pipe.set(STATUS_PREFIX + job_id, "queued", ex=self._ttl)
        pipe.lpush(QUEUE_KEY, job_id)
        pipe.zadd(SUBMITTED_KEY, {job_id: now})
        pipe.zremrangebyscore(SUBMITTED_KEY, 0, now - 2 * self._ttl)
        pipe.execute()

    def claim(self) -> tuple[str, dict[str, Any]] | None:
        """Pop the oldest queued job into processing, or None if empty.

        Jobs whose payload expired mid-queue are dropped and skipped.
        """
        while True:
            raw_id = self._redis.lmove(QUEUE_KEY, PROCESSING_KEY, "RIGHT", "LEFT")
            if raw_id is None:
                return None
            job_id = _text(raw_id)
            raw = self._redis.get(PAYLOAD_PREFIX + job_id)
            if raw is not None:
                break
            self._redis.lrem(PROCESSING_KEY, 1, job_id)
        self._redis.set(STATUS_PREFIX + job_id, "running", ex=self._ttl)
        self._redis.set(CLAIM_PREFIX + job_id, "claimed", ex=self._lease)
        return job_id, json.loads(raw)

    def write_result(self, job_id: str, result: dict[str, Any]) -> None:
        """Store the final result and retire the job from processing."""
        pipe = self._redis.pipeline(transaction=True)
        pipe.set(RESULT_PREFIX + job_id, json.dumps(result), ex=self._ttl)
        pipe.lrem(PROCESSING_KEY, 0, job_id)
        pipe.delete(CLAIM_PREFIX + job_id)
        pipe.delete(STATUS_PREFIX + job_id)
        pipe.execute()

    def get(self, job_id: str) -> JobSnapshot:
        """Read a job's state; a stored result is claimed (deleted) by
        the read itself."""
        raw = self._redis.getdel(RESULT_PREFIX + job_id)
        if raw is not None:
            return JobSnapshot("done", json.loads(raw))
        status = self._redis.get(STATUS_PREFIX + job_id)
        if status is not None:
            return JobSnapshot(_text(status))
        if self._redis.zscore(SUBMITTED_KEY, job_id) is not None:
            return JobSnapshot("gone")
        return JobSnapshot("missing")

    def sweep(self) -> int:
        """Requeue processing jobs whose claim lease expired.

        Returns the number of requeued jobs. A crashed worker's job
        re-runs instead of vanishing.
        """
        requeued = 0
        now = time.time()
        for job_id in self._redis.lrange(PROCESSING_KEY, 0, -1):
            if self._redis.get(CLAIM_PREFIX + _text(job_id)) is not None:
                continue
            self._requeue(_text(job_id), now)
            requeued += 1
        return requeued

    def _requeue(self, job_id: str, now: float) -> None:
        """Move an abandoned processing job back onto the queue."""
        pipe = self._redis.pipeline(transaction=True)
        pipe.lrem(PROCESSING_KEY, 0, job_id)
        pipe.lpush(QUEUE_KEY, job_id)
        pipe.set(STATUS_PREFIX + job_id, "queued", ex=self._ttl)
        pipe.zadd(SUBMITTED_KEY, {job_id: now})
        pipe.execute()
