"""Export the FastAPI OpenAPI schema as JSON (no Redis, no model weights).

Building the app's schema does not start the lifespan, so this runs in CI
and feeds Scalar/Redoc, SDK generators, and contract diffs.

Examples::

    uv run python scripts/export_openapi.py
    uv run python scripts/export_openapi.py docs/openapi.json
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

from blitzid.api._app import create_app

_DEFAULT = Path("docs/openapi.json")


def main(argv: list[str]) -> int:
    target = Path(argv[1]) if len(argv) > 1 else _DEFAULT
    schema = create_app().openapi()
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(schema, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {target} (OpenAPI {schema['openapi']}, {len(schema['paths'])} paths)")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
