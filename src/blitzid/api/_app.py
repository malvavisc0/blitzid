"""FastAPI application: lifespan, engine singletons, workers, routes.

Engine singletons start at lifespan startup (not per request), guarded
by per-engine locks, with the face detector's LRU cache disabled —
uploads are unique images, so the cache gives no hits and its
non-thread-safe ``OrderedDict`` LRU would be a data race. Worker
threads claim queued jobs from Redis and run analyses locally (the
concurrency cap bounds per-process analyses); a lease sweeper requeues
jobs abandoned by a crashed worker.

Env knobs (see ``.env.example``): ``BLITZID_API_REDIS_URL``,
``BLITZID_API_MAX_UPLOAD_MB``, ``BLITZID_API_JOB_TTL_SECONDS``,
``BLITZID_API_MAX_QUEUED_JOBS``, ``BLITZID_API_MAX_CONCURRENT_JOBS``,
``BLITZID_API_JOB_LEASE_SECONDS`` — plus ``BLITZID_MODELS_DIR`` for
the engine weights.
"""

from __future__ import annotations

import base64
import logging
import os
import threading
import uuid
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from dataclasses import dataclass
from typing import Annotated, Any

import redis
from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile

from .. import DocumentCropper, FaceDetectorDNN, MRZReader, RapidOCRReader
from ..exceptions import BlitzIDError
from ._analyze import Engines, run_job
from ._jobs import JobSnapshot, QueueFullError, RedisJobStore
from ._schemas import CropResponse, HealthResponse, JobCreated, JobPayload
from ._upload import encode_jpeg_b64, validate_image_upload

_ANALYSIS_TYPES = frozenset({"face", "ocr", "mrz"})
_SIDES = frozenset({"front", "back", "unknown"})
_LOG = logging.getLogger(__name__)


@dataclass(frozen=True)
class ApiConfig:
    """API env-knob configuration, resolved once per app creation."""

    redis_url: str
    max_upload_bytes: int
    ttl_seconds: int
    max_queued: int
    max_concurrent: int
    lease_seconds: int

    @classmethod
    def from_env(cls) -> ApiConfig:
        """Read the ``BLITZID_API_*`` knobs with their defaults."""
        return cls(
            redis_url=os.environ.get(
                "BLITZID_API_REDIS_URL", "redis://localhost:6379/0"
            ),
            max_upload_bytes=int(os.environ.get("BLITZID_API_MAX_UPLOAD_MB", "20"))
            * 1024
            * 1024,
            ttl_seconds=int(os.environ.get("BLITZID_API_JOB_TTL_SECONDS", "900")),
            max_queued=int(os.environ.get("BLITZID_API_MAX_QUEUED_JOBS", "50")),
            max_concurrent=int(os.environ.get("BLITZID_API_MAX_CONCURRENT_JOBS", "2")),
            lease_seconds=int(os.environ.get("BLITZID_API_JOB_LEASE_SECONDS", "120")),
        )


def _build_engines() -> Engines:
    """Start the engine singletons; unavailable engines degrade to None."""
    try:
        detector = FaceDetectorDNN(log_level=logging.WARNING, enable_cache=False)
    except BlitzIDError as e:
        _LOG.warning("face engine unavailable: %s", e)
        detector = None
    ocr_reader = None
    mrz_reader = None
    try:
        ocr_reader = RapidOCRReader(log_level=logging.WARNING)
        mrz_reader = MRZReader(reader=ocr_reader, log_level=logging.WARNING)
    except BlitzIDError as e:
        _LOG.warning("ocr/mrz engines unavailable: %s", e)
    return Engines(detector=detector, ocr_reader=ocr_reader, mrz_reader=mrz_reader)


class _Workers:
    """Worker threads claiming jobs, plus the processing-lease sweeper."""

    def __init__(
        self,
        store: RedisJobStore,
        engines: Engines,
        config: ApiConfig,
    ) -> None:
        self._store = store
        self._engines = engines
        self._config = config
        self._stop = threading.Event()
        self._threads = [
            threading.Thread(
                target=self._work, name=f"blitzid-worker-{index}", daemon=True
            )
            for index in range(config.max_concurrent)
        ]
        self._sweeper = threading.Thread(
            target=self._sweep, name="blitzid-sweeper", daemon=True
        )

    def start(self) -> None:
        """Recover abandoned jobs, then start workers and the sweeper."""
        self._sweep_once()
        for thread in self._threads:
            thread.start()
        self._sweeper.start()

    def stop(self) -> None:
        """Signal stop and join running analyses (bounded wait)."""
        self._stop.set()
        for thread in [*self._threads, self._sweeper]:
            thread.join(timeout=5.0)

    def _work(self) -> None:
        while not self._stop.is_set():
            claimed = self._claim_safely()
            if claimed is None:
                self._stop.wait(0.1)
                continue
            self._execute(claimed[0], claimed[1])

    def _execute(self, job_id: str, payload: dict[str, Any]) -> None:
        """Run one claimed job; any failure becomes a failed result."""
        try:
            result = run_job(JobPayload.model_validate(payload), self._engines)
        except Exception as e:  # report, never crash the worker
            _LOG.warning("job %s failed: %s: %s", job_id, type(e).__name__, e)
            result = {"state": "failed", "error": f"{type(e).__name__}: {e}"}
        try:
            self._store.write_result(job_id, result)
        except redis.RedisError as e:
            _LOG.error("job %s result lost (store unavailable): %s", job_id, e)

    def _claim_safely(self) -> tuple[str, dict[str, Any]] | None:
        try:
            return self._store.claim()
        except redis.RedisError:
            self._stop.wait(1.0)
            return None

    def _sweep(self) -> None:
        interval = max(1.0, self._config.lease_seconds / 3)
        while not self._stop.wait(interval):
            self._sweep_once()

    def _sweep_once(self) -> None:
        try:
            requeued = self._store.sweep()
        except redis.RedisError as e:
            _LOG.error("lease sweep failed: %s", e)
        else:
            if requeued:
                _LOG.warning("lease sweep requeued %d abandoned job(s)", requeued)


def create_app(
    store: RedisJobStore | None = None, engines: Engines | None = None
) -> FastAPI:
    """Build the FastAPI app.

    Args:
        store: Job store to use; default-constructed from
            ``BLITZID_API_REDIS_URL``. Tests inject a fake.
        engines: Engine singletons to use; default-constructed at
            lifespan startup. Tests inject stubs.
    """

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        config = ApiConfig.from_env()
        job_store = (
            store
            if store is not None
            else RedisJobStore.from_url(
                config.redis_url,
                config.ttl_seconds,
                config.max_queued,
                config.lease_seconds,
            )
        )
        app_engines = engines if engines is not None else _build_engines()
        workers = _Workers(job_store, app_engines, config)
        app.state.config = config
        app.state.store = job_store
        app.state.engines = app_engines
        app.state.cropper = DocumentCropper(detector=app_engines.detector)
        workers.start()
        yield
        workers.stop()

    app = FastAPI(title="blitzid", lifespan=lifespan)
    _register_routes(app)
    return app


def _parse_types(values: list[str]) -> list[str]:
    """Parse the ``types`` form field: repeated and/or comma-separated."""
    flat = [value.strip() for entry in values for value in entry.split(",")]
    types = [value for value in flat if value]
    if not types:
        raise HTTPException(
            422, "at least one analysis type is required: face, ocr, mrz"
        )
    unknown = sorted(set(types) - _ANALYSIS_TYPES)
    if unknown:
        names = ", ".join(unknown)
        raise HTTPException(422, f"unknown analysis type(s): {names}")
    return list(dict.fromkeys(types))


def _engine_missing(analysis_type: str, engines: Engines) -> bool:
    """Whether the engine for one analysis type is unavailable."""
    if analysis_type == "face":
        return engines.detector is None
    return engines.ocr_reader is None


def _ensure_engines(types: list[str], engines: Engines) -> None:
    """Reject analyses whose engine is unavailable (missing extra/weights)."""
    unavailable = sorted(
        type_name for type_name in types if _engine_missing(type_name, engines)
    )
    if unavailable:
        raise HTTPException(
            400,
            f"analysis engine(s) unavailable: {', '.join(unavailable)}; "
            "ocr/mrz need the blitzid[ocr] extra, face needs the SCRFD weights",
        )


def _submit_safely(store: RedisJobStore, job_id: str, payload: dict[str, Any]) -> None:
    """Submit a job, mapping store failures to 503 + Retry-After."""
    try:
        store.submit(job_id, payload)
    except QueueFullError as e:
        raise HTTPException(503, str(e), headers={"Retry-After": "1"}) from e
    except redis.RedisError as e:
        raise HTTPException(
            503, f"job store unavailable: {e}", headers={"Retry-After": "1"}
        ) from e


def _register_routes(app: FastAPI) -> None:
    """Register the /analyze, /jobs, /crop, and /health routes."""

    @app.post("/analyze", status_code=202, response_model=JobCreated)
    def submit_job(
        request: Request,
        image: Annotated[UploadFile, File()],
        types: Annotated[list[str] | None, Form()] = None,
    ) -> JobCreated:
        """Submit an analysis job (async-only queue)."""
        analysis_types = _parse_types(types or [])
        _ensure_engines(analysis_types, request.app.state.engines)
        data = image.file.read()
        validate_image_upload(data, request.app.state.config.max_upload_bytes)
        job_id = uuid.uuid4().hex
        payload = JobPayload(
            image_b64=base64.b64encode(data).decode("ascii"),
            types=analysis_types,  # type: ignore[arg-type]
        )
        _submit_safely(request.app.state.store, job_id, payload.model_dump())
        return JobCreated(job_id=job_id, status="queued", status_url=f"/jobs/{job_id}")

    @app.get("/jobs/{job_id}")
    def get_job(job_id: str, request: Request) -> dict[str, Any]:
        """Poll a job; a done or failed result is claimed by the read."""
        try:
            snapshot = request.app.state.store.get(job_id)
        except redis.RedisError as e:
            raise HTTPException(
                503, f"job store unavailable: {e}", headers={"Retry-After": "1"}
            ) from e
        return _job_response(snapshot)

    @app.post("/crop", response_model=CropResponse)
    def crop_document(
        request: Request,
        image: Annotated[UploadFile, File()],
        side: Annotated[str, Form()] = "unknown",
    ) -> CropResponse:
        """QC a document upload and return the canonical perspective crop."""
        if side not in _SIDES:
            raise HTTPException(422, "side must be one of: front, back, unknown")
        data = image.file.read()
        img = validate_image_upload(data, request.app.state.config.max_upload_bytes)
        crop, report = request.app.state.cropper.crop(img, side)
        return CropResponse(
            quad=[list(point) for point in report.quad] if report.quad else None,
            crop_base64=encode_jpeg_b64(crop) if crop is not None else None,
            width=report.width,
            height=report.height,
            side=side,
            face_found=report.face_found,
            checks=dict(report.checks),
            verdict=report.verdict,
        )

    @app.get("/health", response_model=HealthResponse)
    def health(request: Request) -> HealthResponse:
        """Report engine, Redis, and job availability for orchestration."""
        store: RedisJobStore = request.app.state.store
        engines: Engines = request.app.state.engines
        redis_ok = store.ping()
        return HealthResponse(
            status="ok" if redis_ok else "degraded",
            models={
                "face": engines.detector is not None,
                "ocr": engines.ocr_reader is not None,
                "mrz": engines.mrz_reader is not None,
            },
            redis=redis_ok,
            jobs=store.counts(),
        )


def _job_response(snapshot: JobSnapshot) -> dict[str, Any]:
    """Map a JobSnapshot to the /jobs response, honoring claim-once."""
    if snapshot.status in ("queued", "running"):
        return {"status": snapshot.status}
    if snapshot.status == "done":
        result = snapshot.result or {}
        if result.get("state") == "failed":
            return {"status": "failed", "error": result.get("error", "")}
        return {"status": "done", **{k: v for k, v in result.items() if k != "state"}}
    if snapshot.status == "gone":
        raise HTTPException(410, "job result is gone (already claimed or expired)")
    raise HTTPException(404, "unknown job id")


app = create_app()
