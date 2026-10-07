# ── Builder: install deps and bake model weights ──────────────
FROM python:3.12-slim AS builder
COPY --from=ghcr.io/astral-sh/uv:latest /uv /bin/
WORKDIR /app
COPY README.md pyproject.toml uv.lock ./
COPY src/ src/
COPY scripts/ scripts/
# Every runtime feature ships in the image: the HTTP API (api), face
# detection/verification in the core, OCR, MRZ, PDF417 barcode reading
# (barcode), LLM-backed structured extraction and Langfuse tracing
# (ocr), and the bundled scripts (scripts). Only the dev toolchain
# stays out.
RUN uv sync --frozen --extra api --extra barcode --extra ocr --extra scripts
# Bake every engine's weights (SCRFD, ArcFace, RapidOCR) so cold start
# needs no downloads.
RUN uv run --no-sync python scripts/download_models.py --models-dir /models

# ── Runtime: venv + baked weights, non-root ─────────────────────
FROM python:3.12-slim
ARG BLITZID_BUILD_REVISION
COPY --from=builder /app /app
COPY --from=builder /models /models
ENV BLITZID_BUILD_REVISION=$BLITZID_BUILD_REVISION
ENV BLITZID_MODELS_DIR=/models
ENV PATH=/app/.venv/bin:$PATH
# Scripts reference fixtures via relative "images/" paths, and the
# smoke tests run from the repo root.
WORKDIR /app
RUN useradd --create-home appuser
USER appuser
EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
    CMD ["python", "-c", "import sys, urllib.request; sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:8000/api/health', timeout=3).status == 200 else 1)"]
CMD ["uvicorn", "blitzid.api:app", "--host", "0.0.0.0", "--port", "8000"]
