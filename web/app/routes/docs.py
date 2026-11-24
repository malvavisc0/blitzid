"""The docs page and the OpenAPI schema it renders."""

from __future__ import annotations

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import FileResponse, HTMLResponse

from app.templating import render_page

router = APIRouter()


@router.get("/docs", response_class=HTMLResponse)
def docs(request: Request) -> HTMLResponse:
    """Quickstart and interactive reference, both from the OpenAPI schema."""
    return render_page(request, "docs.html", "docs")


@router.get("/openapi.json")
def openapi(request: Request) -> FileResponse:
    """The schema baked into the image at build time."""
    path = request.app.state.settings.openapi_path
    if not path.is_file():
        raise HTTPException(
            status_code=503,
            detail=(
                "OpenAPI schema is not bundled; run "
                "`uv run python scripts/export_openapi.py "
                "web/app/static/openapi.json`."
            ),
        )
    return FileResponse(
        path,
        media_type="application/json",
        headers={"Cache-Control": "public, max-age=300"},
    )
