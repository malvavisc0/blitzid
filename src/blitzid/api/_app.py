"""FastAPI application: lifespan, engine singletons, workers, routes.

Engine singletons start at lifespan startup (not per request), guarded
by per-engine locks, with the face detector's LRU cache disabled —
uploads are unique images, so the cache gives no hits and its
non-thread-safe ``OrderedDict`` LRU would be a data race. Worker
threads claim queued jobs from Redis and run analyses locally (the
concurrency cap bounds per-process analyses); a heartbeat extends each
running job's claim lease, and a lease sweeper requeues jobs abandoned
by a crashed worker.

Env knobs (see ``.env.example``): ``BLITZID_API_REDIS_URL``,
``BLITZID_API_MAX_UPLOAD_MB``, ``BLITZID_API_JOB_TTL_SECONDS``,
``BLITZID_API_MAX_QUEUED_JOBS``, ``BLITZID_API_MAX_CONCURRENT_JOBS``,
``BLITZID_API_JOB_LEASE_SECONDS`` — plus ``BLITZID_MODELS_DIR`` for
the engine weights and ``BLITZID_LLM_*`` for the structured engine.
"""

from __future__ import annotations

import importlib
import logging
import os
import threading
import time
import uuid
from collections.abc import AsyncIterator, Callable, Iterator, Mapping
from contextlib import asynccontextmanager, contextmanager
from dataclasses import dataclass, replace
from importlib.metadata import version
from types import MappingProxyType
from typing import Annotated, Any

import numpy as np
import redis
from fastapi import (
    Depends,
    FastAPI,
    File,
    Form,
    HTTPException,
    Path,
    Request,
    Response,
    UploadFile,
)
from fastapi.openapi.utils import get_openapi
from fastapi.responses import JSONResponse

from blitzid._image import crop_with_padding
from blitzid.api._analyze import FACE_CROP_PADDING, Engines, run_job
from blitzid.api._auth import SaasError, build_api_auth, require_api_key
from blitzid.api._jobs import (
    ClaimedJob,
    JobSnapshot,
    QueueFullError,
    RedisJobStore,
    StaleClaimError,
)
from blitzid.api._liveness import _CHALLENGE_TTL, ChallengeStore
from blitzid.api._schemas import (
    ChallengeResponse,
    ComparedFace,
    CropResponse,
    FaceRef,
    HealthResponse,
    JobCreated,
    JobResponse,
    LivenessResponse,
    VerifyResponse,
)
from blitzid.api._upload import (
    encode_jpeg_b64,
    validate_analyze_upload,
    validate_image_upload,
)
from blitzid.exceptions import BlitzIDError, FaceVerificationError
from blitzid.face._antispoof import AntiSpoofReader
from blitzid.face._attributes import FaceAttributeReader
from blitzid.face._face import Face
from blitzid.face._motion import verify_action
from blitzid.face.detector import FaceDetectorDNN
from blitzid.face.verifier import FaceVerifier
from blitzid.reading.barcode import BarcodeReader
from blitzid.reading.document import DocumentCropper
from blitzid.reading.mrz import MRZReader
from blitzid.reading.ocr import RapidOCRReader
from blitzid.reading.structurize import StructuredOCRReader

_SIDES = frozenset({"front", "back", "unknown"})
_LOG = logging.getLogger(__name__)

_OPENAPI_TAGS: list[dict[str, str]] = [
    {
        "name": "Jobs",
        "description": "Submit a document or image job and poll for its result.",
    },
    {
        "name": "Documents",
        "description": "Capture QC and perspective cropping for document photos.",
    },
    {"name": "Faces", "description": "1:1 face comparison with evidence."},
    {
        "name": "Liveness",
        "description": "Randomized action challenge and three-frame verification.",
    },
    {
        "name": "Health",
        "description": "Engine, job store, and job availability for orchestration.",
    },
]

_OPENAPI_DESCRIPTION = """\
Analyze identity documents and faces.

## Quickstart

1. **Create an API key** in the dashboard. Self-hosted deployments need no key.
2. **Check the API**

```bash
curl https://api.blitzid.io/api/health \\
  -H "Authorization: Bearer $BLITZID_API_KEY"
```

3. **Submit an analysis** — one image or PDF, one or more types.

```bash
curl https://api.blitzid.io/api/analyze \\
  -F image=@id.jpg \\
  -F types=face,mrz \\
  -H "Authorization: Bearer $BLITZID_API_KEY"
```

4. **Poll the job** — a submit returns `202` with a `status_url`. The first
   finished read returns the result; a second read returns `410`.

```bash
curl https://api.blitzid.io/api/jobs/$JOB_ID \\
  -H "Authorization: Bearer $BLITZID_API_KEY"
```

## Authentication

Hosted requests use `Authorization: Bearer <api key>`. Self-hosted deployments
have no built-in auth; put them behind your own proxy.

## Errors

`400` bad upload or unavailable engine · `401` invalid key · `403` no active
plan · `410` result claimed or expired · `413` upload too large · `415` PDF on
an image-only route · `422` invalid field or no usable face · `429` rate limit ·
`503` queue or job store unavailable (retry after `Retry-After`).

**Decisions stay yours.** Endpoints return signals and scores; choose your own
thresholds and review policy.

[Integration guide](https://github.com/malvavisc0/blitzid/blob/HEAD/docs/integration.md)
"""

_BODY_SCHEMA_RENAMES: dict[str, str] = {
    "Body_submit_job_api_analyze_post": "AnalyzeUpload",
    "Body_crop_document_api_crop_post": "CropUpload",
    "Body_verify_faces_api_verify_post": "VerifyUpload",
    "Body_check_liveness_api_liveness_session_post": "LivenessFrames",
}


class _JobCancelled(Exception):
    """Raised in a worker whose claim on the running job was lost."""


@dataclass(frozen=True)
class ApiConfig:
    """API env-knob configuration, resolved once per app creation."""

    redis_url: str
    max_upload_bytes: int
    ttl_seconds: int
    max_queued: int
    max_concurrent: int
    lease_seconds: int
    saas_database_url: str = ""
    auth_required: bool = False

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
            saas_database_url=os.environ.get("BLITZID_SAAS_DATABASE_URL", ""),
            auth_required=os.environ.get("BLITZID_API_AUTH_REQUIRED", "0").lower()
            in {"1", "true", "yes", "on"},
        )


def _build_engines() -> Engines:
    """Start the engine singletons; unavailable engines degrade to None."""
    try:
        detector = FaceDetectorDNN(log_level=logging.WARNING, enable_cache=False)
    except BlitzIDError as e:
        _LOG.warning("face engine unavailable: %s", e)
        detector = None
    verifier = None
    attribute_reader = None
    antispoof_reader = None
    if detector is not None:
        try:
            verifier = FaceVerifier(detector=detector, log_level=logging.WARNING)
        except BlitzIDError as e:
            _LOG.warning("verification engine unavailable: %s", e)
        try:
            attribute_reader = FaceAttributeReader(
                detector=detector, log_level=logging.WARNING
            )
        except BlitzIDError as e:
            _LOG.warning("attribute engine unavailable: %s", e)
        try:
            antispoof_reader = AntiSpoofReader(
                detector=detector, log_level=logging.WARNING
            )
        except BlitzIDError as e:
            _LOG.warning("antispoof engine unavailable: %s", e)
    ocr_reader = None
    mrz_reader = None
    try:
        ocr_reader = RapidOCRReader(log_level=logging.WARNING)
        mrz_reader = MRZReader(reader=ocr_reader, log_level=logging.WARNING)
    except BlitzIDError as e:
        _LOG.warning("ocr/mrz engines unavailable: %s", e)
    barcode_reader = None
    try:
        barcode_reader = BarcodeReader(log_level=logging.WARNING)
    except BlitzIDError as e:
        _LOG.warning("barcode engine unavailable: %s", e)
    structured_reader = None
    try:
        importlib.import_module("pydantic_ai")
        structured_reader = StructuredOCRReader(log_level=logging.WARNING)
    except (ImportError, BlitzIDError) as e:
        _LOG.warning("structured engine unavailable: %s", e)
    return Engines(
        detector=detector,
        ocr_reader=ocr_reader,
        mrz_reader=mrz_reader,
        barcode_reader=barcode_reader,
        verifier=verifier,
        structured_reader=structured_reader,
        attribute_reader=attribute_reader,
        antispoof_reader=antispoof_reader,
    )


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
            self._execute(claimed)

    def _execute(self, claimed: ClaimedJob) -> None:
        """Run one claimed job; any failure becomes a failed result.

        The claim lease is extended by a heartbeat thread while the
        analysis runs, so the sweeper only requeues jobs whose worker
        actually died. When the lease is lost mid-run (the job was
        re-claimed by another worker), this worker finishes what it
        is doing but discards the result — the new owner's result
        wins.
        """
        cancelled = threading.Event()
        try:
            with self._lease_heartbeat(claimed, cancelled):
                result = run_job(claimed.image, claimed.types, self._engines)
        except _JobCancelled:
            _LOG.warning(
                "job %s abandoned by this worker (claim lost mid-run)",
                claimed.job_id,
            )
            return
        except Exception as e:  # report, never crash the worker
            _LOG.warning("job %s failed: %s: %s", claimed.job_id, type(e).__name__, e)
            result = {"state": "failed", "error": f"{type(e).__name__}: {e}"}
        try:
            self._store.write_result(claimed.job_id, claimed.token, result)
        except StaleClaimError:
            _LOG.warning(
                "job %s result discarded (claim lost before the write)",
                claimed.job_id,
            )
        except redis.RedisError as e:
            _LOG.error("job %s result lost (store unavailable): %s", claimed.job_id, e)

    @contextmanager
    def _lease_heartbeat(
        self, claimed: ClaimedJob, cancelled: threading.Event
    ) -> Iterator[None]:
        """Extend a job's claim lease on a background thread.

        Sets *cancelled* when the claim lease is lost (the running
        analysis is not interrupted); the check after the context
        exits raises :class:`_JobCancelled` so the worker discards
        its result.
        """
        stop = threading.Event()
        heartbeat = threading.Thread(
            target=self._heartbeat,
            args=(claimed, stop, cancelled),
            name=f"blitzid-heartbeat-{claimed.job_id}",
            daemon=True,
        )
        heartbeat.start()
        try:
            yield
        finally:
            stop.set()
            heartbeat.join(timeout=1.0)
        if cancelled.is_set():
            raise _JobCancelled

    def _heartbeat(
        self, claimed: ClaimedJob, stop: threading.Event, cancelled: threading.Event
    ) -> None:
        """Renew a job's claim lease until told to stop.

        A failed renewal (the claim was lost) flags the result for
        discard — the re-claimed worker owns the job now.
        """
        interval = max(1.0, self._config.lease_seconds / 3)
        while not stop.wait(interval):
            try:
                if not self._store.renew(claimed.job_id, claimed.token):
                    cancelled.set()
                    return
            except redis.RedisError as e:
                _LOG.warning("job %s lease renewal failed: %s", claimed.job_id, e)

    def _claim_safely(self) -> ClaimedJob | None:
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
            recovered = self._store.sweep()
        except redis.RedisError as e:
            _LOG.error("lease sweep failed: %s", e)
        else:
            if recovered:
                _LOG.warning("lease sweep recovered %d abandoned job(s)", recovered)


def _rename_schema_refs(node: Any, renames: Mapping[str, str]) -> None:
    """Rewrite component names inside one OpenAPI document."""
    if isinstance(node, dict):
        ref = node.get("$ref")
        prefix = "#/components/schemas/"
        if isinstance(ref, str) and ref.startswith(prefix):
            name = ref.removeprefix(prefix)
            if name in renames:
                node["$ref"] = f"{prefix}{renames[name]}"
        for value in node.values():
            _rename_schema_refs(value, renames)
    elif isinstance(node, list):
        for item in node:
            _rename_schema_refs(item, renames)


def _openapi_schema(app: FastAPI) -> dict[str, Any]:
    """Build the OpenAPI schema with auth, links, and friendly body names."""
    if app.openapi_schema is not None:
        return app.openapi_schema
    schema = get_openapi(
        title=app.title,
        version=app.version,
        openapi_version=app.openapi_version,
        description=app.description,
        routes=app.routes,
        tags=app.openapi_tags,
        servers=app.servers,
        terms_of_service=app.terms_of_service,
        contact=app.contact,
        license_info=app.license_info,
        separate_input_output_schemas=app.separate_input_output_schemas,
    )
    components: dict[str, Any] = schema.setdefault("components", {})
    components.setdefault("securitySchemes", {})["bearerAuth"] = {
        "type": "http",
        "scheme": "bearer",
        "description": (
            "Hosted API key (`Authorization: Bearer $BLITZID_API_KEY`). "
            "Self-hosted deployments need it only when a proxy enforces auth."
        ),
    }
    # The empty alternative documents self-hosted calls without a key.
    schema["security"] = [{"bearerAuth": []}, {}]
    schemas: dict[str, Any] = components.get("schemas", {})
    for old, new in _BODY_SCHEMA_RENAMES.items():
        if old in schemas:
            schemas[new] = schemas.pop(old)
            schemas[new]["title"] = new
    _rename_schema_refs(schema, _BODY_SCHEMA_RENAMES)
    app.openapi_schema = schema
    return schema


def create_app(
    store: RedisJobStore | None = None, engines: Engines | None = None
) -> FastAPI:
    """Build the FastAPI app with every route under the ``/api`` prefix.

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
        app.state.challenges = ChallengeStore()
        app.state.cropper = DocumentCropper(
            detector=app_engines.detector,
            detector_lock=app_engines.detector_lock,
        )
        api_auth = (
            build_api_auth(config.saas_database_url, config.redis_url)
            if config.auth_required
            else None
        )
        app.state.api_auth = api_auth
        workers.start()
        yield
        workers.stop()
        if api_auth is not None:
            api_auth.close()

    app = FastAPI(
        title="BlitzID API",
        version=version("blitzid"),
        description=_OPENAPI_DESCRIPTION,
        openapi_tags=_OPENAPI_TAGS,
        servers=[
            {"url": "https://api.blitzid.io", "description": "Hosted API"},
            {
                "url": "http://localhost:8000",
                "description": "Self-hosted deployment (default port)",
            },
        ],
        contact={
            "name": "BlitzID",
            "url": "https://blitzid.io",
            "email": "hello@blitzid.io",
        },
        license_info={
            "name": "MIT",
            "url": "https://github.com/malvavisc0/blitzid/blob/HEAD/LICENSE",
        },
        lifespan=lifespan,
    )
    # FastAPI types openapi() as a plain method; replacing it caches the
    # decorated schema on this app instance.
    app.openapi = lambda: _openapi_schema(app)  # type: ignore[method-assign]

    @app.exception_handler(SaasError)
    async def _saas_error(request: Request, exc: SaasError) -> JSONResponse:
        headers = (
            {"Retry-After": str(exc.retry_after)}
            if exc.retry_after is not None
            else None
        )
        return JSONResponse(
            status_code=exc.status_code,
            content={"error": {"code": exc.code, "message": exc.message}},
            headers=headers,
        )

    app.state.revision = os.environ.get("BLITZID_BUILD_REVISION") or None
    _register_routes(app)
    return app


def _parse_types(values: list[str]) -> list[str]:
    """Parse the ``types`` form field: repeated and/or comma-separated."""
    flat = [value.strip() for entry in values for value in entry.split(",")]
    types = [value for value in flat if value]
    if not types:
        raise HTTPException(
            422,
            f"at least one analysis type is required: {', '.join(_ENGINE_CHECKS)}",
        )
    unknown = sorted(set(types) - _ENGINE_CHECKS.keys())
    if unknown:
        names = ", ".join(unknown)
        raise HTTPException(422, f"unknown analysis type(s): {names}")
    return list(dict.fromkeys(types))


def _parse_bbox(value: str | None, field: str) -> tuple[int, int, int, int] | None:
    """Parse an optional ``x,y,w,h`` form field (None when omitted).

    Raises:
        HTTPException: ``422`` for a malformed or degenerate box.
    """
    if value is None or not value.strip():
        return None
    try:
        x, y, w, h = (int(part) for part in value.split(","))
    except ValueError as e:
        raise HTTPException(422, f"{field} must be 'x,y,w,h' integers") from e
    if w <= 0 or h <= 0:
        raise HTTPException(422, f"{field} width and height must be positive")
    return (x, y, w, h)


def _overlap(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> int:
    """Intersection area of two ``(x, y, w, h)`` boxes."""
    ax, ay, aw, ah = a
    bx, by, bw, bh = b
    width = max(0, min(ax + aw, bx + bw) - max(ax, bx))
    height = max(0, min(ay + ah, by + bh) - max(ay, by))
    return width * height


def _pin_score(face: Face, pinned: tuple[int, int, int, int]) -> tuple[float, float]:
    """Rank pin coverage, then IoU, without favoring large peripheral faces."""
    intersection = _overlap(face.bbox, pinned)
    if intersection == 0:
        return 0.0, 0.0
    face_area = face.bbox[2] * face.bbox[3]
    pinned_area = pinned[2] * pinned[3]
    union = face_area + pinned_area - intersection
    return intersection / min(face_area, pinned_area), intersection / union


def _pick_face(
    faces: list[Face], pinned: tuple[int, int, int, int] | None, image: str
) -> Face:
    """Pick the compared face: best overlap with a pinned box, else best score.

    Raises:
        FaceVerificationError: When the image holds no usable face.
    """
    if not faces:
        raise FaceVerificationError(f"No face detected in {image}")
    if pinned is None:
        return max(faces, key=lambda f: f.confidence)
    best = max(faces, key=lambda f: _pin_score(f, pinned))
    if _overlap(best.bbox, pinned) <= 0:
        raise FaceVerificationError(f"No face found in the pinned region of {image}")
    return best


def _face_evidence(img: np.ndarray, selected: Face, faces: list[Face]) -> ComparedFace:
    """Build the /verify evidence for one compared face."""
    crop = crop_with_padding(img, selected.bbox, FACE_CROP_PADDING)
    return ComparedFace(
        bbox=list(selected.bbox),
        confidence=selected.confidence,
        crop_base64=encode_jpeg_b64(crop),
        alternatives=[
            FaceRef(bbox=list(face.bbox), confidence=face.confidence)
            for face in faces
            if face is not selected
        ],
    )


_ENGINE_CHECKS: Mapping[str, Callable[[Engines], bool]] = MappingProxyType(
    {
        "face": lambda engines: engines.detector is None,
        "attributes": lambda engines: engines.attribute_reader is None,
        "antispoof": lambda engines: engines.antispoof_reader is None,
        "ocr": lambda engines: engines.ocr_reader is None,
        "mrz": lambda engines: engines.mrz_reader is None,
        "barcode": lambda engines: engines.barcode_reader is None,
        "structured": lambda engines: (
            engines.structured_reader is None or engines.ocr_reader is None
        ),
        "consistency": lambda engines: (
            engines.mrz_reader is None
            or engines.structured_reader is None
            or engines.ocr_reader is None
        ),
    }
)


def _ensure_engines(types: list[str], engines: Engines) -> None:
    """Reject analyses whose engine is unavailable (missing weights/zxing)."""
    unavailable = sorted(
        type_name for type_name in types if _ENGINE_CHECKS[type_name](engines)
    )
    if unavailable:
        raise HTTPException(
            400,
            f"analysis engine(s) unavailable: {', '.join(unavailable)}; "
            "ocr/mrz need the RapidOCR engine, barcode the zxing-cpp engine, "
            "face/attributes/antispoof need the YuNet weights "
            "(attributes also FairFace, antispoof also MiniFASNet), and "
            "structured/consistency need the OCR engine plus BLITZID_LLM_* "
            "config",
        )


def _submit_safely(
    store: RedisJobStore, job_id: str, image: bytes, types: list[str]
) -> None:
    """Submit a job, mapping store failures to 503 + Retry-After."""
    try:
        store.submit(job_id, image, types)
    except QueueFullError as e:
        raise HTTPException(503, str(e), headers={"Retry-After": "1"}) from e
    except redis.RedisError as e:
        raise HTTPException(
            503, f"job store unavailable: {e}", headers={"Retry-After": "1"}
        ) from e


def _register_routes(app: FastAPI) -> None:
    """Register the /api/analyze, /api/jobs, /api/crop, /api/verify,
    /api/liveness, and /api/health routes."""

    @app.post(
        "/api/analyze",
        dependencies=[Depends(require_api_key)],
        openapi_extra={"security": [{"bearerAuth": []}]},
        status_code=202,
        response_model=JobCreated,
        tags=["Jobs"],
        summary="Submit an analysis job",
        description=(
            "Upload one image or PDF and request one or more analysis types. "
            "The job is queued; poll `status_url` until `status` is `done` or "
            "`failed`. A finished result is claim-once, so a second read "
            "returns `410`.\n\n"
            "Analysis types: `face`, `attributes`, `antispoof`, `ocr`, `mrz`, "
            "`barcode`, `structured`, `consistency`."
        ),
        responses={
            400: {
                "description": (
                    "Rejected upload (undecodable or out of bounds) or an "
                    "unavailable analysis engine."
                )
            },
            413: {"description": "Upload exceeds the configured size cap."},
            422: {"description": "Missing, unknown, or malformed analysis type."},
            503: {
                "description": "Job queue full or job store unavailable.",
                "headers": {
                    "Retry-After": {
                        "description": "Seconds to wait before retrying.",
                        "schema": {"type": "string"},
                    }
                },
            },
        },
    )
    def submit_job(
        request: Request,
        image: Annotated[UploadFile, File(description="Image or PDF to analyze.")],
        types: Annotated[
            list[str] | None,
            Form(
                description=(
                    "Analysis types, repeated or comma-separated (e.g. `ocr,mrz`)."
                ),
                examples=[["ocr", "mrz"]],
            ),
        ] = None,
    ) -> JobCreated:
        """Submit an analysis job (async-only queue)."""
        analysis_types = _parse_types(types or [])
        _ensure_engines(analysis_types, request.app.state.engines)
        max_bytes = request.app.state.config.max_upload_bytes
        data = image.file.read(max_bytes + 1)
        validate_analyze_upload(data, max_bytes)
        job_id = uuid.uuid4().hex
        _submit_safely(request.app.state.store, job_id, data, analysis_types)
        return JobCreated(
            job_id=job_id, status="queued", status_url=f"/api/jobs/{job_id}"
        )

    @app.get(
        "/api/jobs/{job_id}",
        dependencies=[Depends(require_api_key)],
        openapi_extra={"security": [{"bearerAuth": []}]},
        response_model=None,
        tags=["Jobs"],
        summary="Get a job result",
        description=(
            "Returns `queued` or `running` while the job is pending. A "
            "finished job is claim-once: the first read returns its result "
            "with `status: done` (or `failed`), and later reads return `410`; "
            "unknown ids return `404`."
        ),
        responses={
            200: {
                "model": JobResponse,
                "description": "Pending status, the claimed result, or a failure.",
            },
            404: {"description": "Unknown job id."},
            410: {"description": "Result already claimed or expired."},
        },
    )
    def get_job(
        job_id: Annotated[
            str, Path(description="Job id returned by `POST /api/analyze`.")
        ],
        request: Request,
    ) -> dict[str, Any]:
        """Poll a job; a done or failed result is claimed by the read."""
        try:
            snapshot = request.app.state.store.get(job_id)
        except redis.RedisError as e:
            raise HTTPException(
                503, f"job store unavailable: {e}", headers={"Retry-After": "1"}
            ) from e
        return _job_response(snapshot)

    @app.post(
        "/api/crop",
        dependencies=[Depends(require_api_key)],
        openapi_extra={"security": [{"bearerAuth": []}]},
        response_model=CropResponse,
        tags=["Documents"],
        summary="Crop a document capture",
        description=(
            "Run capture QC and return the canonical perspective crop, the "
            "detected quad, and per-check verdicts."
        ),
        responses={
            400: {"description": "Undecodable or out-of-bounds image."},
            413: {"description": "Upload exceeds the configured size cap."},
            415: {"description": "PDF upload; this endpoint accepts images only."},
            422: {"description": "Unknown `side` value."},
        },
    )
    def crop_document(
        request: Request,
        image: Annotated[UploadFile, File(description="Document photo to crop.")],
        side: Annotated[
            str,
            Form(description="Document side: `front`, `back`, or `unknown`."),
        ] = "unknown",
    ) -> CropResponse:
        """QC a document upload and return the canonical perspective crop."""
        if side not in _SIDES:
            raise HTTPException(422, "side must be one of: front, back, unknown")
        max_bytes = request.app.state.config.max_upload_bytes
        data = image.file.read(max_bytes + 1)
        img = validate_image_upload(data, max_bytes)
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

    @app.post(
        "/api/verify",
        dependencies=[Depends(require_api_key)],
        openapi_extra={"security": [{"bearerAuth": []}]},
        response_model=VerifyResponse,
        tags=["Faces"],
        summary="Compare two faces",
        description=(
            "Detect one face in each image and compare them 1:1. Optionally "
            "pin a face with `x,y,w,h` and override the decision threshold."
        ),
        responses={
            400: {"description": "Face verification engine unavailable."},
            413: {"description": "An upload exceeds the configured size cap."},
            415: {"description": "PDF upload; this endpoint accepts images only."},
            422: {
                "description": (
                    "Invalid threshold or bbox, or no usable face in an image."
                )
            },
        },
    )
    def verify_faces(
        request: Request,
        image1: Annotated[
            UploadFile, File(description="First image (e.g. the selfie).")
        ],
        image2: Annotated[
            UploadFile, File(description="Second image (e.g. the document portrait).")
        ],
        threshold: Annotated[
            float | None,
            Form(
                description=(
                    "Override the backend threshold; `verified` becomes "
                    "`similarity >= threshold`."
                )
            ),
        ] = None,
        face1_bbox: Annotated[
            str | None,
            Form(description="Pin a face in `image1` as `x,y,w,h` pixels."),
        ] = None,
        face2_bbox: Annotated[
            str | None,
            Form(description="Pin a face in `image2` as `x,y,w,h` pixels."),
        ] = None,
    ) -> VerifyResponse:
        """Compare one face from each image (synchronous)."""
        engines: Engines = request.app.state.engines
        if engines.verifier is None or engines.detector is None:
            raise HTTPException(
                400,
                "face verification engine unavailable (YuNet detector and "
                "SFace weights are required)",
            )
        if threshold is not None and not -1.0 <= threshold <= 1.0:
            raise HTTPException(422, "threshold must be between -1.0 and 1.0")
        pinned1 = _parse_bbox(face1_bbox, "face1_bbox")
        pinned2 = _parse_bbox(face2_bbox, "face2_bbox")
        config: ApiConfig = request.app.state.config
        img1 = validate_image_upload(
            image1.file.read(config.max_upload_bytes + 1), config.max_upload_bytes
        )
        img2 = validate_image_upload(
            image2.file.read(config.max_upload_bytes + 1), config.max_upload_bytes
        )
        with engines.detector_lock:
            faces1 = engines.detector.detect_face_landmarks(img1)
            faces2 = engines.detector.detect_face_landmarks(img2)
        try:
            selected1 = _pick_face(faces1, pinned1, "image1")
            selected2 = _pick_face(faces2, pinned2, "image2")
            with engines.verifier_lock:
                result = engines.verifier.verify_faces(img1, selected1, img2, selected2)
        except FaceVerificationError as e:
            raise HTTPException(422, str(e)) from e
        if threshold is not None:
            result = replace(
                result, threshold=threshold, verified=result.similarity >= threshold
            )
        return VerifyResponse(
            verified=result.verified,
            similarity=result.similarity,
            threshold=result.threshold,
            face1=_face_evidence(img1, selected1, faces1),
            face2=_face_evidence(img2, selected2, faces2),
            processing_time_ms=round(result.processing_time * 1000, 1),
            backend=result.backend,
        )

    @app.post(
        "/api/liveness/challenge",
        dependencies=[Depends(require_api_key)],
        openapi_extra={"security": [{"bearerAuth": []}]},
        response_model=ChallengeResponse,
        tags=["Liveness"],
        summary="Issue a liveness challenge",
        description=(
            "Issue a one-use, short-lived random action for the client to "
            "perform while capturing the three frames."
        ),
    )
    def issue_liveness_challenge(request: Request) -> ChallengeResponse:
        """Issue one random liveness action (one-use, short-lived)."""
        challenge_id, action = request.app.state.challenges.issue()
        return ChallengeResponse(
            challenge_id=challenge_id, action=action, expires_in=_CHALLENGE_TTL
        )

    @app.post(
        "/api/liveness/session",
        dependencies=[Depends(require_api_key)],
        openapi_extra={"security": [{"bearerAuth": []}]},
        response_model=LivenessResponse,
        tags=["Liveness"],
        summary="Check a liveness session",
        description=(
            "Consume the challenge and verify the requested action across "
            "three frames. Challenges are one-use: an unknown, expired, or "
            "replayed id returns `410`."
        ),
        responses={
            400: {"description": "Face engine unavailable."},
            410: {"description": "Challenge unknown, expired, or already used."},
            413: {"description": "A frame exceeds the configured size cap."},
            415: {"description": "PDF upload; this endpoint accepts images only."},
            422: {"description": "No usable face in a frame."},
        },
    )
    def check_liveness(
        request: Request,
        challenge_id: Annotated[
            str,
            Form(description="Challenge id from `POST /api/liveness/challenge`."),
        ],
        frame1: Annotated[UploadFile, File(description="Capture frame 1 of 3.")],
        frame2: Annotated[UploadFile, File(description="Capture frame 2 of 3.")],
        frame3: Annotated[UploadFile, File(description="Capture frame 3 of 3.")],
    ) -> LivenessResponse:
        """Verify one random action across three capture frames."""
        engines: Engines = request.app.state.engines
        if engines.detector is None:
            raise HTTPException(
                400, "liveness requires the YuNet detector (face engine)"
            )
        action = request.app.state.challenges.consume(challenge_id)
        if action is None:
            raise HTTPException(410, "challenge unknown, expired, or already used")
        config: ApiConfig = request.app.state.config
        frames = [
            validate_image_upload(
                upload.file.read(config.max_upload_bytes + 1), config.max_upload_bytes
            )
            for upload in (frame1, frame2, frame3)
        ]
        start = time.perf_counter()
        with engines.detector_lock:
            found = [engines.detector.detect_face_landmarks(frame) for frame in frames]
        try:
            picked = [
                _pick_face(faces, None, f"frame {index + 1}")
                for index, faces in enumerate(found)
            ]
        except FaceVerificationError as e:
            raise HTTPException(422, str(e)) from e
        evidence = verify_action(action, picked)
        return LivenessResponse(
            live=evidence.verified,
            action=action,
            nose_shift=evidence.nose_shift,
            mouth_ratio_change=evidence.mouth_ratio_change,
            size_ratio=evidence.size_ratio,
            processing_time_ms=round((time.perf_counter() - start) * 1000, 1),
        )

    @app.get(
        "/api/health",
        response_model=HealthResponse,
        tags=["Health"],
        summary="Service health",
        responses={
            503: {
                "model": HealthResponse,
                "description": (
                    "Job store unreachable; the service cannot accept or process jobs."
                ),
            }
        },
    )
    def health(request: Request, response: Response) -> HealthResponse:
        """Report engine, Redis, and job availability for orchestration.

        Returns ``503`` when the job store is unreachable (the service
        cannot accept or process jobs, so the container is unhealthy);
        a missing engine stays ``200`` with its ``models`` flag false —
        submitting that analysis yields a precise ``400``.
        """
        store: RedisJobStore = request.app.state.store
        engines: Engines = request.app.state.engines
        config: ApiConfig = request.app.state.config
        redis_ok = store.ping()
        auth_ok: bool | None = None
        if config.auth_required:
            resolver = getattr(request.app.state, "api_auth", None)
            auth_ok = bool(resolver is not None and resolver.ready())
        healthy = redis_ok and auth_ok is not False
        if not healthy:
            response.status_code = 503
        return HealthResponse(
            status="ok" if healthy else "degraded",
            version=request.app.version,
            revision=request.app.state.revision,
            models={
                "face": engines.detector is not None,
                "verify": engines.verifier is not None,
                "attributes": engines.attribute_reader is not None,
                "antispoof": engines.antispoof_reader is not None,
                "ocr": engines.ocr_reader is not None,
                "mrz": engines.mrz_reader is not None,
                "barcode": engines.barcode_reader is not None,
                "structured": engines.structured_reader is not None,
            },
            redis=redis_ok,
            jobs=store.counts(),
            auth=auth_ok,
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
