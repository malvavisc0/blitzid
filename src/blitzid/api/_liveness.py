"""Short-lived liveness challenges (claim-once, RAM only).

A challenge is one random action with a one-use id. The session
endpoint consumes it before checking anything, so a response can never
be replayed for a second signup. Challenges live in process memory:
the server runs one process (threads inside), and they are 120-second
ceremonies, not records.
"""

from __future__ import annotations

import secrets
import threading
import time
import uuid

from blitzid.face._motion import LivenessAction

__all__ = ["ChallengeStore"]

_ACTIONS: tuple[LivenessAction, ...] = ("turn_head", "smile", "move_closer")
_CHALLENGE_TTL = 120.0


class ChallengeStore:
    """Issues and consumes one-use, 120-second liveness challenges."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._open: dict[str, tuple[LivenessAction, float]] = {}

    def issue(self) -> tuple[str, LivenessAction]:
        """Issue a fresh challenge, returning its id and action."""
        challenge_id = uuid.uuid4().hex
        action = _ACTIONS[secrets.randbelow(len(_ACTIONS))]
        with self._lock:
            self._sweep()
            self._open[challenge_id] = (action, time.monotonic() + _CHALLENGE_TTL)
        return challenge_id, action

    def consume(self, challenge_id: str) -> LivenessAction | None:
        """Claim one challenge, or None when unknown, used, or expired."""
        with self._lock:
            self._sweep()
            entry = self._open.pop(challenge_id, None)
        return entry[0] if entry else None

    def _sweep(self) -> None:
        """Drop expired challenges (call with the lock held)."""
        now = time.monotonic()
        stale = [key for key, (_, due) in self._open.items() if due <= now]
        for key in stale:
            del self._open[key]
