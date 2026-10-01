"""Langfuse tracing for the LLM structured-extraction agent.

Strictly opt-in via the standard Langfuse SDK environment variables:
``LANGFUSE_PUBLIC_KEY`` / ``LANGFUSE_SECRET_KEY`` (and optionally
``LANGFUSE_BASE_URL``). Without them nothing is initialized and no
spans leave the process. When set, the Langfuse client installs its
span processor on the process's OpenTelemetry tracer provider —
creating one (``service.name`` = "blitzid" via ``OTEL_SERVICE_NAME``)
when none is installed, attaching to a host application's provider
otherwise — and the pydantic-ai agent emits spans for every model call.

Fail-soft: a missing ``langfuse`` install or a broken credential
setup warns and disables tracing — observability can never take the
reader down.

PII caveat: exported spans carry the OCR text fed to the model —
names, document numbers, and dates from the analyzed document. Point
``LANGFUSE_BASE_URL`` at a Langfuse instance you control before
enabling tracing on real documents.
"""

from __future__ import annotations

import atexit
import functools
import logging
import os
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from langfuse import Langfuse

__all__ = ["langfuse_tracing"]

logger = logging.getLogger(__name__)

LANGFUSE_PUBLIC_KEY_ENV = "LANGFUSE_PUBLIC_KEY"
LANGFUSE_SECRET_KEY_ENV = "LANGFUSE_SECRET_KEY"
LANGFUSE_BASE_URL_ENV = "LANGFUSE_BASE_URL"


@functools.cache
def langfuse_tracing() -> Langfuse | None:
    """Initialise Langfuse tracing once per process.

    Returns:
        The shared Langfuse client, or None while tracing is disabled
        (missing credentials, missing ``langfuse`` install, or a
        failed initialization — the latter logged as a warning).

    Idempotent: the first call decides, later calls return the same
    client.
    """
    public_key = os.environ.get(LANGFUSE_PUBLIC_KEY_ENV, "").strip()
    secret_key = os.environ.get(LANGFUSE_SECRET_KEY_ENV, "").strip()
    if not public_key or not secret_key:
        return None
    try:
        from langfuse import get_client

        os.environ.setdefault("OTEL_SERVICE_NAME", "blitzid")
        client = get_client()
    except Exception as e:
        logger.warning("Langfuse tracing unavailable: %s", e)
        return None
    atexit.register(_flush, client)
    logger.info(
        "Langfuse tracing enabled (base URL: %s)",
        os.environ.get(LANGFUSE_BASE_URL_ENV, "https://cloud.langfuse.com"),
    )
    return client


def _flush(client: Langfuse) -> None:
    """Flush buffered spans; safe at process exit."""
    try:
        client.flush()
    except Exception:
        logger.warning("Failed to flush Langfuse spans")
