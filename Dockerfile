# ── Builder: install deps and bake model weights ──────────────
FROM python:3.12-slim AS builder
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/
WORKDIR /app
COPY README.md pyproject.toml uv.lock ./
COPY src/ src/
COPY scripts/ scripts/
RUN uv sync --frozen --no-dev --extra api --extra ocr
RUN uv run --no-sync python scripts/download_models.py --models-dir /models

# ── Runtime: venv + baked weights, non-root ─────────────────────
FROM python:3.12-slim
COPY --from=builder /app /app
COPY --from=builder /models /models
ENV BLITZID_MODELS_DIR=/models
ENV PATH=/app/.venv/bin:$PATH
RUN useradd --create-home appuser
USER appuser
EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
    CMD ["python", "-c", "import sys, urllib.request; sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:8000/health', timeout=3).status == 200 else 1)"]
CMD ["uvicorn", "blitzid.api:app", "--host", "0.0.0.0", "--port", "8000"]
