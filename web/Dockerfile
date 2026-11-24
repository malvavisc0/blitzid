# syntax=docker/dockerfile:1.7

# ── Schema: install the API project and export its OpenAPI document ──
# Only the schema is needed, so the lifespan (and model weights) never
# starts. The API release and the docs therefore come from one source.
FROM python:3.12-slim AS schema
COPY --from=ghcr.io/astral-sh/uv:latest /uv /bin/
WORKDIR /src
COPY README.md LICENSE THIRD_PARTY_NOTICES.md pyproject.toml uv.lock ./
COPY LICENSES/ LICENSES/
COPY src/ src/
COPY scripts/export_openapi.py scripts/export_openapi.py
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --extra api
RUN uv run --no-sync python scripts/export_openapi.py /openapi.json

# ── Runtime: web dependencies, templates, and static assets ──────────
FROM python:3.12-slim AS runtime
COPY --from=ghcr.io/astral-sh/uv:latest /uv /bin/
WORKDIR /app
ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    PYTHONUNBUFFERED=1
COPY web/pyproject.toml web/uv.lock ./
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-install-project --no-dev
COPY web/app ./app
COPY web/alembic.ini ./
COPY web/migrations ./migrations
COPY --from=schema /openapi.json ./app/static/openapi.json

# Pin the reference bundle; it is served locally, no runtime CDN request.
ARG SCALAR_VERSION=1.83.0
RUN python -c "import urllib.request; urllib.request.urlretrieve( \
        'https://cdn.jsdelivr.net/npm/@scalar/api-reference@${SCALAR_VERSION}/dist/browser/standalone.js', \
        '/app/app/static/scalar.js')"

ARG BLITZID_WEB_VERSION=dev
ENV BLITZID_WEB_VERSION=${BLITZID_WEB_VERSION} \
    PATH="/app/.venv/bin:${PATH}"
RUN useradd --create-home appuser && chown -R appuser:appuser /app
USER appuser

EXPOSE 8080
HEALTHCHECK --interval=30s --timeout=3s --start-period=10s --retries=3 \
    CMD ["python", "-c", "import sys, urllib.request; sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:8080/healthz', timeout=3).status == 200 else 1)"]

CMD ["sh", "-c", "alembic upgrade head && exec uvicorn app.main:app --host 0.0.0.0 --port 8080 --proxy-headers --forwarded-allow-ips '*' --no-server-header"]
