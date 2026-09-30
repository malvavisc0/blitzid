"""Redis-backed job store for the blitzid API.

Jobs live entirely in Redis memory (the server runs with persistence
disabled): a queue list, a processing list, a job hash per submitted
job (raw image bytes + requested types), a token-owned claim-lease key
per running job, a result string key per finished job, and a
submission registry zset that distinguishes expired/claimed jobs
(``410``) from unknown ids (``404``). Every key carries an ``EXPIRE``
bounded by the job TTL, and the registry zsets are pruned to a grace
window (the results index prunes itself on every write), so memory
stays bounded — nothing ever reaches disk.

Claim protocol: ``LMOVE`` pops a job id from the queue into the
processing list and the claiming worker immediately takes the claim
key with ``SET NX`` holding a unique token — exactly one worker can
own a job at a time, even when the sweeper requeues an id that was
mid-claim. Results are written once and read via ``GETDEL``
(claim-once — every subsequent read sees the id as gone). A running
worker extends its claim lease (and status TTL) via
:meth:`RedisJobStore.renew`, which only succeeds for the token owner;
a worker whose lease was lost learns this from ``renew`` and abandons
its result. A worker crash is recovered by the lease sweeper: ids
still in processing past ``BLITZID_API_JOB_LEASE_SECONDS`` are pushed
back onto the queue (respecting the queue cap; abandoned ids whose
payload expired are dropped instead).
"""

from __future__ import annotations

import json
import time
import uuid
from dataclasses import dataclass
from typing import Any, cast

import redis

from ..exceptions import BlitzIDError

QUEUE_KEY = "blitzid:queue"
PROCESSING_KEY = "blitzid:processing"
JOB_PREFIX = "blitzid:job:"
STATUS_PREFIX = "blitzid:status:"
RESULT_PREFIX = "blitzid:result:"
CLAIM_PREFIX = "blitzid:claim:"
SUBMITTED_KEY = "blitzid:submitted"
RESULTS_KEY = "blitzid:results"


class QueueFullError(BlitzIDError):
    """The queue already holds the maximum number of jobs."""


class StaleClaimError(BlitzIDError):
    """The claim on a job was lost; the calling worker's result is
    discarded because a re-claimed worker owns the job."""


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


@dataclass(frozen=True)
class ClaimedJob:
    """One job claimed by a worker, with its payload and claim token.

    Attributes:
        job_id: The job identifier.
        image: The uploaded image's raw bytes.
        types: The requested analysis types.
        token: The unique claim token; the worker must pass it to
            :meth:`RedisJobStore.renew` and
            :meth:`RedisJobStore.write_result`, which only succeed for
            the current claim owner.
    """

    job_id: str
    image: bytes
    types: list[str]
    token: str


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
            is requeued by :meth:`sweep` unless :meth:`renew` extends
            it.
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
        """Queue, processing, and unclaimed-result counts for /health.

        Unclaimed results are tracked in a zset scored by their expiry
        timestamp (pruned here and on every result write), so counting
        stays O(log n) instead of scanning the keyspace.
        """
        try:
            now = time.time()
            pipe = self._redis.pipeline(transaction=False)
            self._prune_results(pipe, now)
            pipe.zcard(RESULTS_KEY)
            pipe.llen(QUEUE_KEY)
            pipe.llen(PROCESSING_KEY)
            _, stored, queued, running = pipe.execute()
            return {
                "queued": int(queued),
                "running": int(running),
                "stored": int(stored),
            }
        except redis.RedisError:
            return {"queued": 0, "running": 0, "stored": 0}

    def submit(self, job_id: str, image: bytes, types: list[str]) -> None:
        """Store the job payload (raw image bytes) and enqueue the id.

        The capacity check and the enqueue run inside one watched
        transaction, so concurrent submissions cannot push past the
        queue cap.

        Raises:
            QueueFullError: If the queue is at capacity.
            redis.RedisError: If Redis is unreachable.
        """
        now = time.time()
        with self._redis.pipeline() as pipe:
            while True:
                try:
                    pipe.watch(QUEUE_KEY)  # type: ignore[no-untyped-call]
                    if int(pipe.llen(QUEUE_KEY)) >= self._max_queued:
                        pipe.unwatch()
                        raise QueueFullError(
                            f"queue is full ({self._max_queued} queued jobs)"
                        )
                    pipe.multi()
                    pipe.hset(
                        JOB_PREFIX + job_id,
                        mapping={"image": image, "types": json.dumps(types)},
                    )
                    pipe.expire(JOB_PREFIX + job_id, self._ttl)
                    pipe.set(STATUS_PREFIX + job_id, "queued", ex=self._ttl)
                    pipe.lpush(QUEUE_KEY, job_id)
                    pipe.zadd(SUBMITTED_KEY, {job_id: now})
                    self._prune_registry(pipe, now)
                    pipe.execute()
                    return
                except redis.WatchError:
                    continue

    def claim(self) -> ClaimedJob | None:
        """Pop the oldest queued job into processing, or None if empty.

        Jobs whose payload expired mid-queue are dropped and skipped.
        The claim key is taken with ``SET NX`` holding a unique token;
        when the sweeper requeued this id in the pop→claim window, the
        claim fails and the id is left for its new queue turn.

        A claimed job holds a token-owned claim-lease key; the worker
        must call :meth:`renew` while it runs or the sweeper requeues
        the job once the lease expires.
        """
        token = uuid.uuid4().hex
        while True:
            raw_id = self._redis.lmove(QUEUE_KEY, PROCESSING_KEY, "RIGHT", "LEFT")
            if raw_id is None:
                return None
            job_id = _text(raw_id)
            raw = self._redis.hgetall(JOB_PREFIX + job_id)
            if not raw:
                self._redis.lrem(PROCESSING_KEY, 1, job_id)
                continue
            claim_key = CLAIM_PREFIX + job_id
            with self._redis.pipeline() as pipe:
                while True:
                    try:
                        pipe.watch(claim_key)  # type: ignore[no-untyped-call]
                        if pipe.get(claim_key) is not None:
                            # requeued by the sweeper mid-claim (or a
                            # live claim survived): leave the id queued
                            # for its new owner's queue turn
                            pipe.unwatch()
                            self._redis.lrem(PROCESSING_KEY, 1, job_id)
                            break
                        pipe.multi()
                        pipe.set(claim_key, token, ex=self._lease)
                        pipe.set(STATUS_PREFIX + job_id, "running", ex=self._ttl)
                        pipe.execute()
                        return ClaimedJob(
                            job_id=job_id,
                            image=cast(bytes, raw[b"image"]),
                            types=json.loads(_text(raw[b"types"])),
                            token=token,
                        )
                    except redis.WatchError:
                        continue
            # a claim or requeue raced us — try the next job

    def renew(self, job_id: str, token: str) -> bool:
        """Extend the claim lease and status TTL (worker heartbeat).

        Returns whether *token* still owns the claim. Renewal keeps
        long analyses from being requeued as abandoned while they run;
        a ``False`` return means the lease was lost (worker crash
        recovery took over) and the worker must abandon its result.
        """
        # compare-and-expire: only the current owner extends the lease
        if self._redis.get(CLAIM_PREFIX + job_id) != token.encode():
            return False
        pipe = self._redis.pipeline(transaction=True)
        pipe.expire(CLAIM_PREFIX + job_id, self._lease)
        pipe.expire(STATUS_PREFIX + job_id, self._ttl)
        return bool(all(pipe.execute()))

    def write_result(self, job_id: str, token: str, result: dict[str, Any]) -> None:
        """Store the final result and retire the job from processing.

        The write runs in a watched transaction guarded by the claim
        token: a worker whose lease was lost (the claim now holds a
        different token or vanished) discards its result with a
        :class:`StaleClaimError` instead of clobbering the re-run's.
        """
        claim_key = CLAIM_PREFIX + job_id
        with self._redis.pipeline() as pipe:
            while True:
                try:
                    pipe.watch(claim_key)  # type: ignore[no-untyped-call]
                    if pipe.get(claim_key) != token.encode():
                        pipe.unwatch()
                        raise StaleClaimError(
                            f"claim on job {job_id} was lost; result discarded"
                        )
                    now = time.time()
                    pipe.multi()
                    pipe.set(RESULT_PREFIX + job_id, json.dumps(result), ex=self._ttl)
                    pipe.zadd(RESULTS_KEY, {job_id: now + self._ttl})
                    self._prune_results(pipe, now)
                    pipe.lrem(PROCESSING_KEY, 0, job_id)
                    pipe.delete(
                        CLAIM_PREFIX + job_id,
                        STATUS_PREFIX + job_id,
                        JOB_PREFIX + job_id,
                    )
                    pipe.execute()
                    return
                except redis.WatchError:
                    continue

    def get(self, job_id: str) -> JobSnapshot:
        """Read a job's state; a stored result is claimed (deleted) by
        the read itself."""
        raw = self._redis.getdel(RESULT_PREFIX + job_id)
        if raw is not None:
            self._redis.zrem(RESULTS_KEY, job_id)
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
        re-runs instead of vanishing. A requeue is guarded by a watch
        on the claim key, so a claim arriving mid-sweep aborts it; it
        also respects the queue cap (an abandoned id is dropped when
        the queue is full and its payload has expired).
        """
        requeued = 0
        now = time.time()
        for raw_id in self._redis.lrange(PROCESSING_KEY, 0, -1):
            if self._requeue(_text(raw_id), now):
                requeued += 1
        return requeued

    def _requeue(self, job_id: str, now: float) -> bool:
        """Move an abandoned processing job back onto the queue.

        Runs inside a watched transaction on the claim key and returns
        False when the claim appears (or exists) — the job is not
        abandoned after all. When the queue is at capacity, the id is
        dropped only if its payload expired; a still-live payload keeps
        the id queued beyond the cap (requeues are bounded by the
        processing list length).
        """
        claim_key = CLAIM_PREFIX + job_id
        with self._redis.pipeline() as pipe:
            while True:
                try:
                    pipe.watch(claim_key, QUEUE_KEY)  # type: ignore[no-untyped-call]
                    if pipe.get(claim_key) is not None:
                        pipe.unwatch()
                        return False
                    queue_full = int(pipe.llen(QUEUE_KEY)) >= self._max_queued
                    payload_expired = int(pipe.hlen(JOB_PREFIX + job_id)) == 0
                    pipe.unwatch()
                    if queue_full:
                        if payload_expired:
                            # nothing left to run: retire the id instead
                            # of pushing the queue past its cap
                            self._redis.lrem(PROCESSING_KEY, 1, job_id)
                            self._redis.delete(STATUS_PREFIX + job_id)
                            return True
                        return False
                    pipe.multi()
                    pipe.lrem(PROCESSING_KEY, 0, job_id)
                    pipe.lpush(QUEUE_KEY, job_id)
                    pipe.set(STATUS_PREFIX + job_id, "queued", ex=self._ttl)
                    pipe.zadd(SUBMITTED_KEY, {job_id: now})
                    self._prune_registry(pipe, now)
                    pipe.execute()
                    return True
                except redis.WatchError:
                    continue

    def _prune_registry(self, pipe: Any, now: float) -> None:
        """Prune the submission registry to its 2x-TTL grace window."""
        pipe.zremrangebyscore(SUBMITTED_KEY, 0, now - 2 * self._ttl)

    def _prune_results(self, pipe: Any, now: float) -> None:
        """Prune the results index to entries not yet expired."""
        pipe.zremrangebyscore(RESULTS_KEY, 0, now)
