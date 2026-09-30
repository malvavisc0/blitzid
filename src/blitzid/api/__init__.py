"""blitzid HTTP API — exposes ``app`` (``uvicorn blitzid.api:app``).

This is the one deliberate exception to the "no re-exports in
subpackage ``__init__``" rule: the ASGI entry point must be importable
as ``blitzid.api:app``. Requires the ``api`` extra
(``pip install blitzid[api]``).
"""

from ._app import app

__all__ = ["app"]
