"""Tests for the blitzid API (gated on fastapi).

App tests run against an in-process job-store fake and stub engines —
no Redis server, no models, no network. Store-logic tests run the real
RedisJobStore against fakeredis (no Redis server). Real-engine
variants run behind BLITZID_OCR_INTEGRATION=1 (downloads models).
"""

from __future__ import annotations

import base64
import logging
import os
import time
import uuid
from collections import deque
from collections.abc import Callable
from pathlib import Path
from threading import Barrier, Event, Thread
from typing import Any

import cv2
import numpy as np
import pytest

from blitzid import (
    AntiSpoofResult,
    ExtractedField,
    Face,
    FaceAttributes,
    FaceDetectorDNN,
    FaceVerificationError,
    ModelError,
    MRZError,
    MRZReader,
    MRZRecord,
    OCRText,
    RapidOCRReader,
    StructuredOCR,
    VerificationResult,
)

pytest.importorskip("fastapi")

import fakeredis
import redis
from fastapi.testclient import TestClient

from blitzid.api._analyze import Engines, _printed_texts
from blitzid.api._app import ApiConfig, _Workers, create_app
from blitzid.api._jobs import (
    CLAIM_PREFIX,
    JOB_PREFIX,
    PROCESSING_KEY,
    QUEUE_KEY,
    RESULT_PREFIX,
    ClaimedJob,
    JobSnapshot,
    QueueFullError,
    RedisJobStore,
    StaleClaimError,
)


def _now() -> float:
    """Monotonic-ish clock the fake can be wound forward with."""
    return time.monotonic()


class MemoryJobStore:
    """RedisJobStore-shaped in-process fake (same interface).

    Mirrors the real store's lease and registry semantics: claims are
    token-owned with expiry timestamps, results are claim-once, and
    the submission registry prunes to a 2x-TTL grace window so expired
    ids become ``missing`` (404), not ``gone`` (410), after the window.
    """

    def __init__(
        self, max_queued: int = 50, ttl_seconds: int = 900, lease_seconds: int = 120
    ) -> None:
        self._queue: deque[str] = deque()
        self._processing: list[str] = []
        self._jobs: dict[str, tuple[bytes, list[str]]] = {}
        self._status: dict[str, str] = {}
        self._results: dict[str, dict[str, Any]] = {}
        self._claims: dict[str, tuple[str, float]] = {}
        self._submitted: dict[str, float] = {}
        self._max_queued = max_queued
        self._ttl = ttl_seconds
        self._lease = lease_seconds

    def ping(self) -> bool:
        return True

    def counts(self) -> dict[str, int]:
        return {
            "queued": len(self._queue),
            "running": len(self._processing),
            "stored": len(self._results),
        }

    def submit(self, job_id: str, image: bytes, types: list[str]) -> None:
        if len(self._queue) >= self._max_queued:
            raise QueueFullError(f"queue is full ({self._max_queued} queued jobs)")
        self._prune_registry()
        self._jobs[job_id] = (image, types)
        self._status[job_id] = "queued"
        self._queue.appendleft(job_id)
        self._submitted[job_id] = _now()

    def claim(self) -> ClaimedJob | None:
        while self._queue:
            job_id = self._queue.pop()
            job = self._jobs.get(job_id)
            if job is None:
                self._status.pop(job_id, None)
                continue
            if job_id in self._claims:
                continue
            token = uuid.uuid4().hex
            self._processing.append(job_id)
            self._status[job_id] = "running"
            self._claims[job_id] = (token, _now() + self._lease)
            return ClaimedJob(job_id=job_id, image=job[0], types=job[1], token=token)
        return None

    def renew(self, job_id: str, token: str) -> bool:
        claim = self._claims.get(job_id)
        if claim is None or claim[0] != token or claim[1] <= _now():
            return False
        self._claims[job_id] = (token, _now() + self._lease)
        return True

    def write_result(self, job_id: str, token: str, result: dict[str, Any]) -> None:
        claim = self._claims.get(job_id)
        if claim is None or claim[0] != token:
            raise StaleClaimError(f"claim on job {job_id} was lost")
        self._results[job_id] = result
        if job_id in self._processing:
            self._processing.remove(job_id)
        self._claims.pop(job_id, None)
        self._jobs.pop(job_id, None)
        self._status.pop(job_id, None)

    def get(self, job_id: str) -> JobSnapshot:
        self._prune_registry()
        result = self._results.pop(job_id, None)
        if result is not None:
            return JobSnapshot("done", result)
        status = self._status.get(job_id)
        if status is not None:
            return JobSnapshot(status)
        if job_id in self._submitted:
            return JobSnapshot("gone")
        return JobSnapshot("missing")

    def sweep(self) -> int:
        requeued = 0
        for job_id in list(self._processing):
            claim = self._claims.get(job_id)
            if claim is not None and claim[1] > _now():
                continue
            self._claims.pop(job_id, None)
            if self._requeue(job_id):
                requeued += 1
        return requeued

    def _requeue(self, job_id: str) -> bool:
        if job_id not in self._jobs:
            self._processing.remove(job_id)
            self._status.pop(job_id, None)
            return True
        if len(self._queue) >= self._max_queued:
            return False
        self._processing.remove(job_id)
        self._queue.appendleft(job_id)
        self._status[job_id] = "queued"
        return True

    def _prune_registry(self) -> None:
        """Drop submitted ids past the 2x-TTL grace window (404)."""
        cutoff = _now() - 2 * self._ttl
        for job_id, submitted_at in list(self._submitted.items()):
            if submitted_at <= cutoff:
                del self._submitted[job_id]


class _StubDetector:
    """FaceDetectorDNN stand-in returning a fixed result or error."""

    def __init__(self, faces: list[Any] | Exception) -> None:
        self._faces = faces
        self.calls = 0

    def detect_face_landmarks(self, image_input: Any) -> list[Any]:
        self.calls += 1
        if isinstance(self._faces, Exception):
            raise self._faces
        return self._faces


class _StubOCRReader:
    """RapidOCRReader stand-in returning fixed lines or an error."""

    def __init__(self, texts: list[OCRText] | Exception) -> None:
        self._texts = texts
        self.calls = 0

    def read(self, image_input: Any) -> list[OCRText]:
        self.calls += 1
        if isinstance(self._texts, Exception):
            raise self._texts
        return self._texts


class _StubMRZReader:
    """MRZReader stand-in returning a fixed record or error."""

    def __init__(self, record: MRZRecord | Exception) -> None:
        self._record = record
        self.calls = 0

    def parse(self, lines: Any) -> MRZRecord:
        self.calls += 1
        if isinstance(self._record, Exception):
            raise self._record
        return self._record


class _StubAttributeReader:
    """FaceAttributeReader stand-in returning a fixed result or error."""

    def __init__(self, attributes: list[Any] | Exception) -> None:
        self._attributes = attributes
        self.calls = 0

    def read_faces(self, img: Any, faces: Any) -> list[Any]:
        self.calls += 1
        if isinstance(self._attributes, Exception):
            raise self._attributes
        return self._attributes


class _StubAntiSpoofReader:
    """AntiSpoofReader stand-in returning fixed scores or an error."""

    def __init__(self, scores: list[Any] | Exception) -> None:
        self._scores = scores
        self.calls = 0

    def read_faces(self, img: Any, faces: Any) -> list[Any]:
        self.calls += 1
        if isinstance(self._scores, Exception):
            raise self._scores
        return self._scores


class _SequenceDetector(_StubDetector):
    """Detector stub returning one face list per call, frame by frame."""

    def __init__(self, batches: list[list[Any]]) -> None:
        super().__init__([_face()])
        self._batches = list(batches)

    def detect_face_landmarks(self, image_input: Any) -> list[Any]:
        return self._batches.pop(0)


class _StubVerifier:
    """FaceVerifier stand-in recording the faces it is asked to compare."""

    def __init__(self, result: VerificationResult | Exception) -> None:
        self._result = result
        self.compared: list[tuple[Any, Any]] = []

    def verify_faces(
        self, image1: Any, face1: Any, image2: Any, face2: Any
    ) -> VerificationResult:
        self.compared.append((face1, face2))
        if isinstance(self._result, Exception):
            raise self._result
        return self._result


class _StubStructuredReader:
    """StructuredOCRReader stand-in returning a fixed record or error."""

    def __init__(self, record: StructuredOCR | Exception) -> None:
        self._record = record
        self.calls = 0

    def read(self, texts: Any, kind: str = "auto") -> StructuredOCR:
        self.calls += 1
        if isinstance(self._record, Exception):
            raise self._record
        return self._record


def _face() -> Face:
    return Face(
        bbox=(10, 10, 60, 60),
        confidence=0.9,
        landmarks=((1, 2), (3, 4), (5, 6), (7, 8), (9, 10)),
    )


def _ghost_face() -> Face:
    return Face(
        bbox=(200, 200, 30, 30),
        confidence=0.6,
        landmarks=((1, 2), (3, 4), (5, 6), (7, 8), (9, 10)),
    )


def _ocr_text() -> OCRText:
    return OCRText(bbox=(0, 0, 20, 8), text="ID", confidence=0.91)


def _mrz_record() -> MRZRecord:
    return MRZRecord(
        mrz_type="TD1",
        document_code="I<",
        issuer="NLD",
        document_number="123456789",
        birth_date="940120",
        sex="M",
        expiry_date="260120",
        nationality="NLD",
        surname="SPECIMEN",
        given_names="VORBEILD",
        optional_data1="",
        optional_data2="",
    )


def _structured_record() -> StructuredOCR:
    return StructuredOCR(
        document_type="id",
        fields=[ExtractedField(name="surname", value="MUSTERMANN", confidence=0.9)],
        raw_text="ID",
    )


def _consistent_record() -> StructuredOCR:
    """Printed side matching the MRZ stub record field for field."""
    return StructuredOCR(
        document_type="id",
        fields=[
            ExtractedField(name="surname", value="SPECIMEN", confidence=0.9),
            ExtractedField(name="given_names", value="VORBEILD", confidence=0.9),
            ExtractedField(name="document_number", value="123456789", confidence=0.9),
            ExtractedField(name="date_of_birth", value="1994-01-20", confidence=0.9),
            ExtractedField(name="date_of_expiry", value="2026-01-20", confidence=0.9),
            ExtractedField(name="sex", value="M", confidence=0.9),
        ],
        raw_text="ID",
    )


def _verification() -> VerificationResult:
    return VerificationResult(
        verified=True,
        similarity=0.97,
        threshold=0.4,
        processing_time=0.123,
        backend="ONNXRuntime",
    )


def _face_attributes() -> Any:
    return FaceAttributes(
        bbox=(10, 10, 60, 60),
        confidence=0.9,
        age="30-39",
        gender="Male",
        race="White",
        age_confidence=0.6,
        gender_confidence=0.95,
        race_confidence=0.7,
    )


def _zoom_face(height: int) -> Face:
    return Face(
        bbox=(30, 10, 40, height),
        confidence=0.9,
        landmarks=((40, 40), (60, 40), (50, 55), (40, 70), (60, 70)),
    )


def _spoof_result() -> Any:
    return AntiSpoofResult(
        bbox=(10, 10, 60, 60),
        confidence=0.9,
        live_score=0.93,
        paper_score=0.05,
        screen_score=0.02,
    )


def _frames() -> dict[str, tuple[str, bytes, str]]:
    return {
        name: (f"{name}.png", _tiny_image_bytes(), "image/png")
        for name in ("frame1", "frame2", "frame3")
    }


def _stub_engines(
    faces: list[Any] | Exception | None = None,
    ocr_texts: list[OCRText] | Exception | None = None,
    mrz: MRZRecord | Exception | None = None,
    verification: VerificationResult | Exception | None = None,
    structured: StructuredOCR | Exception | None = None,
    attributes: list[Any] | Exception | None = None,
    antispoof: list[Any] | Exception | None = None,
) -> Engines:
    return Engines(
        detector=_StubDetector([_face()] if faces is None else faces),
        ocr_reader=_StubOCRReader([_ocr_text()] if ocr_texts is None else ocr_texts),
        mrz_reader=_StubMRZReader(_mrz_record() if mrz is None else mrz),
        verifier=_StubVerifier(
            _verification() if verification is None else verification
        ),
        structured_reader=_StubStructuredReader(
            _structured_record() if structured is None else structured
        ),
        attribute_reader=_StubAttributeReader(
            [_face_attributes()] if attributes is None else attributes
        ),
        antispoof_reader=_StubAntiSpoofReader(
            [_spoof_result()] if antispoof is None else antispoof
        ),
    )


class _BrokenStore(MemoryJobStore):
    """Store fake simulating an unreachable Redis."""

    def ping(self) -> bool:
        return False

    def submit(self, job_id: str, image: bytes, types: list[str]) -> None:
        raise redis.ConnectionError("Redis is down")


def _image_bytes(img: np.ndarray, ext: str = ".png") -> bytes:
    ok, buffer = cv2.imencode(ext, img)
    assert ok
    return buffer.tobytes()


def _tiny_image_bytes() -> bytes:
    return _image_bytes(np.zeros((60, 80, 3), dtype=np.uint8))


def _client(
    store: MemoryJobStore | RedisJobStore | None = None,
    engines: Engines | None = None,
) -> TestClient:
    app = create_app(store=store, engines=engines)
    return TestClient(app)


_API = "/api"


def _env_client(
    monkeypatch: pytest.MonkeyPatch,
    store: MemoryJobStore | RedisJobStore | None = None,
    engines: Engines | None = None,
    **env: str,
) -> TestClient:
    """Build a TestClient with BLITZID_API_* env knobs set for its lifespan."""
    for key, value in env.items():
        monkeypatch.setenv(key, value)
    return _client(store, engines)


def _submit(
    client: TestClient,
    types: str | list[str] = "face",
    data: bytes | None = None,
) -> Any:
    return client.post(
        f"{_API}/analyze",
        files={"image": ("doc.png", data or _tiny_image_bytes(), "image/png")},
        data={"types": types},
    )


def _poll_done(client: TestClient, job_id: str, timeout: float = 5.0) -> dict[str, Any]:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        response = client.get(f"{_API}/jobs/{job_id}")
        assert response.status_code == 200
        body = response.json()
        if body.get("status") in ("done", "failed"):
            return body
        time.sleep(0.02)
    pytest.fail("job did not complete in time")


def _decode_base64_jpeg(data: str) -> np.ndarray:
    img = cv2.imdecode(
        np.frombuffer(base64.b64decode(data), np.uint8), cv2.IMREAD_COLOR
    )
    assert img is not None
    return img


class TestSubmitLifecycle:
    def test_submit_and_poll_face_job(self) -> None:
        with _client(store=MemoryJobStore(), engines=_stub_engines()) as client:
            response = _submit(client)
            assert response.status_code == 202
            body = response.json()
            assert body["status"] == "queued"
            assert body["status_url"] == f"/api/jobs/{body['job_id']}"
            result = _poll_done(client, body["job_id"])
            assert result["status"] == "done"
            face = result["face"]
            assert len(face["faces"]) == 1
            assert face["faces"][0]["bbox"] == [10, 10, 60, 60]
            assert face["faces"][0]["landmarks"] == [
                [1, 2],
                [3, 4],
                [5, 6],
                [7, 8],
                [9, 10],
            ]
            assert face["processing_time_ms"] >= 0
            crop = _decode_base64_jpeg(face["faces"][0]["crop_base64"])
            assert crop.shape[2] == 3

    def test_comma_separated_and_repeated_types(self) -> None:
        store = MemoryJobStore()
        with _client(store=store, engines=_stub_engines()) as client:
            response = _submit(client, types=["face,ocr", "face"])
            assert response.status_code == 202
            result = _poll_done(client, response.json()["job_id"])
            assert set(result) == {"status", "face", "ocr"}
            assert result["ocr"]["lines"][0]["text"] == "ID"

    def test_claim_once_second_read_gone(self) -> None:
        with _client(store=MemoryJobStore(), engines=_stub_engines()) as client:
            response = _submit(client)
            job_id = response.json()["job_id"]
            assert _poll_done(client, job_id)["status"] == "done"
            assert client.get(f"{_API}/jobs/{job_id}").status_code == 410

    def test_unknown_job_id_404(self) -> None:
        with _client(store=MemoryJobStore()) as client:
            response = client.get(f"{_API}/jobs/doesnotexist")
            assert response.status_code == 404

    def test_failed_job_reports_error(self) -> None:
        engines = _stub_engines(faces=RuntimeError("engine exploded"))
        with _client(store=MemoryJobStore(), engines=engines) as client:
            response = _submit(client)
            result = _poll_done(client, response.json()["job_id"])
            assert result["status"] == "failed"
            assert "engine exploded" in result["error"]

    def test_mrz_not_found_is_section_error(self) -> None:
        engines = _stub_engines(mrz=MRZError("no MRZ found among OCR text lines"))
        with _client(store=MemoryJobStore(), engines=engines) as client:
            response = _submit(client, types="face,mrz")
            result = _poll_done(client, response.json()["job_id"])
            assert result["status"] == "done"
            assert "no MRZ found" in result["mrz"]["error"]
            assert result["face"]["faces"]

    def test_ocr_reader_error_is_section_error(self) -> None:
        engines = _stub_engines(ocr_texts=MRZError("reader broke"))
        with _client(store=MemoryJobStore(), engines=engines) as client:
            response = _submit(client, types="ocr")
            result = _poll_done(client, response.json()["job_id"])
            assert result["status"] == "done"
            assert "reader broke" in result["ocr"]["error"]

    def test_ttl_expiry_returns_410(self, monkeypatch: pytest.MonkeyPatch) -> None:
        store = RedisJobStore(
            fakeredis.FakeStrictRedis(), ttl_seconds=1, max_queued=50, lease_seconds=120
        )
        client = _env_client(
            monkeypatch, store=store, BLITZID_API_MAX_CONCURRENT_JOBS="0"
        )
        with client:
            response = _submit(client)
            assert response.status_code == 202
            job_id = response.json()["job_id"]
            assert client.get(f"{_API}/jobs/{job_id}").json()["status"] == "queued"
            time.sleep(1.3)
            response = client.get(f"{_API}/jobs/{job_id}")
            assert response.status_code == 410

    def test_lease_sweeper_requeues_abandoned_job(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        store = RedisJobStore(
            fakeredis.FakeStrictRedis(),
            ttl_seconds=900,
            max_queued=50,
            lease_seconds=120,
        )
        client = _env_client(
            monkeypatch, store=store, BLITZID_API_MAX_CONCURRENT_JOBS="0"
        )
        with client:
            _submit(client)
            claimed = store.claim()
            assert claimed is not None and claimed.job_id
            store._redis.delete(CLAIM_PREFIX + claimed.job_id)
            assert store.sweep() == 1
            snapshot = store.get(claimed.job_id)
            assert snapshot.status == "queued"
            requeued = store.claim()
            assert requeued is not None and requeued.job_id == claimed.job_id

    def test_lease_sweeper_keeps_live_claims(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        store = RedisJobStore(
            fakeredis.FakeStrictRedis(),
            ttl_seconds=900,
            max_queued=50,
            lease_seconds=120,
        )
        client = _env_client(
            monkeypatch, store=store, BLITZID_API_MAX_CONCURRENT_JOBS="0"
        )
        with client:
            _submit(client)
            store.claim()
            assert store.sweep() == 0

    def test_running_job_lease_is_renewed(self) -> None:
        """A heartbeat thread extends the claim while a job runs, so a
        long analysis is not requeued as abandoned."""
        renewed: list[str] = []

        class _CountingStore(MemoryJobStore):
            def renew(self, job_id: str, token: str) -> bool:
                renewed.append(job_id)
                return super().renew(job_id, token)

        store = _CountingStore(lease_seconds=3)
        config = ApiConfig(
            redis_url="redis://localhost:6379/0",
            max_upload_bytes=20 * 1024 * 1024,
            ttl_seconds=900,
            max_queued=50,
            max_concurrent=0,
            lease_seconds=3,
        )
        workers = _Workers(store, _stub_engines(), config)
        store.submit("job", b"image", ["face"])
        claimed = store.claim()
        assert claimed is not None
        with workers._lease_heartbeat(claimed, Event()):
            time.sleep(2.0)
        assert renewed.count("job") >= 1
        # the lease survived past the original expiry
        assert store.renew("job", claimed.token) is True

    def test_worker_abandons_job_when_lease_is_lost(self) -> None:
        """A worker whose renewal fails (lost claim) discards its
        result instead of clobbering the re-claimed worker's."""
        store = RedisJobStore(
            fakeredis.FakeStrictRedis(),
            ttl_seconds=900,
            max_queued=50,
            lease_seconds=120,
        )
        config = ApiConfig(
            redis_url="redis://localhost:6379/0",
            max_upload_bytes=20 * 1024 * 1024,
            ttl_seconds=900,
            max_queued=50,
            max_concurrent=1,
            lease_seconds=120,
        )
        engines = _stub_engines()
        workers = _Workers(store, engines, config)
        store.submit("job", _tiny_image_bytes(), ["face"])
        claimed = store.claim()
        assert claimed is not None
        # a re-claimed worker took over the job mid-run
        store._redis.set(CLAIM_PREFIX + "job", "other-token", ex=120)
        with pytest.raises(StaleClaimError):
            workers._store.write_result("job", claimed.token, {"state": "done"})
        assert store._redis.get(CLAIM_PREFIX + "job") == b"other-token"

    def test_execute_discards_result_when_lease_lost(self) -> None:
        """_execute on a job whose claim is lost mid-run writes no
        result — the re-claimed worker owns the job."""

        class _SlowDetector(_StubDetector):
            def detect_face_landmarks(self, image_input: Any) -> list[Any]:
                time.sleep(2.0)  # outlast a heartbeat interval
                return super().detect_face_landmarks(image_input)

        store = RedisJobStore(
            fakeredis.FakeStrictRedis(),
            ttl_seconds=900,
            max_queued=50,
            lease_seconds=3,
        )
        config = ApiConfig(
            redis_url="redis://localhost:6379/0",
            max_upload_bytes=20 * 1024 * 1024,
            ttl_seconds=900,
            max_queued=50,
            max_concurrent=1,
            lease_seconds=3,
        )
        engines = _stub_engines()
        engines.detector = _SlowDetector([_face()])
        workers = _Workers(store, engines, config)
        store.submit("job", _tiny_image_bytes(), ["face"])
        claimed = store.claim()
        assert claimed is not None
        # a re-claimed worker took over the job mid-run
        store._redis.set(CLAIM_PREFIX + "job", "other-token", ex=3)
        workers._execute(claimed)
        assert store._redis.get(RESULT_PREFIX + "job") is None
        assert store._redis.get(CLAIM_PREFIX + "job") == b"other-token"


class TestSubmitValidation:
    def test_unknown_type_422(self) -> None:
        with _client(store=MemoryJobStore()) as client:
            assert _submit(client, types="lipsum").status_code == 422

    def test_missing_types_422(self) -> None:
        with _client(store=MemoryJobStore()) as client:
            response = client.post(
                f"{_API}/analyze",
                files={"image": ("d.png", _tiny_image_bytes(), "image/png")},
            )
            assert response.status_code == 422

    def test_missing_image_422(self) -> None:
        with _client(store=MemoryJobStore()) as client:
            response = client.post(f"{_API}/analyze", data={"types": "face"})
            assert response.status_code == 422

    def test_corrupt_bytes_400(self) -> None:
        with _client(store=MemoryJobStore()) as client:
            response = _submit(client, data=b"\x00\x01\x02not-an-image")
            assert response.status_code == 400

    def test_pdf_bytes_415(self) -> None:
        with _client(store=MemoryJobStore()) as client:
            response = _submit(client, data=b"%PDF-1.4 fake letter page")
            assert response.status_code == 415
            assert "jpg/png/webp/bmp/tiff" in response.json()["detail"]

    def test_oversized_upload_413(self, monkeypatch: pytest.MonkeyPatch) -> None:
        big = b"\xff" * (1024 * 1024 + 1)
        with _env_client(
            monkeypatch,
            store=MemoryJobStore(),
            BLITZID_API_MAX_UPLOAD_MB="1",
        ) as client:
            response = _submit(client, data=big)
            assert response.status_code == 413

    def test_oversized_dimensions_400(self) -> None:
        huge = _image_bytes(np.zeros((10001, 32, 3), dtype=np.uint8))
        with _client(store=MemoryJobStore(), engines=_stub_engines()) as client:
            assert _submit(client, data=huge).status_code == 400

    def test_queue_full_503(self, monkeypatch: pytest.MonkeyPatch) -> None:
        store = MemoryJobStore(max_queued=1)
        with _env_client(
            monkeypatch,
            store=store,
            BLITZID_API_MAX_CONCURRENT_JOBS="0",
        ) as client:
            assert _submit(client).status_code == 202
            response = _submit(client)
            assert response.status_code == 503
            assert response.headers["Retry-After"] == "1"

    def test_redis_down_503(self) -> None:
        with _client(store=_BrokenStore()) as client:
            response = _submit(client)
            assert response.status_code == 503
            assert response.headers["Retry-After"] == "1"

    def test_engine_unavailable_400(self) -> None:
        engines = Engines(detector=None, ocr_reader=None, mrz_reader=None)
        with _client(store=MemoryJobStore(), engines=engines) as client:
            response = _submit(client, types="face")
            assert response.status_code == 400
            response = _submit(client, types="ocr,mrz")
            assert response.status_code == 400

    def test_mrz_engine_unavailable_400(self) -> None:
        engines = Engines(
            detector=None,
            ocr_reader=_StubOCRReader([_ocr_text()]),
            mrz_reader=None,
        )
        with _client(store=MemoryJobStore(), engines=engines) as client:
            assert _submit(client, types="mrz").status_code == 400
            assert _submit(client, types="ocr").status_code == 202


class TestHealth:
    def test_reports_engines_redis_and_jobs(self) -> None:
        with _client(store=MemoryJobStore(), engines=_stub_engines()) as client:
            response = client.get(f"{_API}/health")
            assert response.status_code == 200
            body = response.json()
            assert body["status"] == "ok"
            assert body["models"] == {
                "face": True,
                "verify": True,
                "attributes": True,
                "antispoof": True,
                "ocr": True,
                "mrz": True,
                "structured": True,
            }
            assert body["redis"] is True
            assert set(body["jobs"]) == {"queued", "running", "stored"}

    def test_degraded_without_engines_and_redis(self) -> None:
        engines = Engines(detector=None, ocr_reader=None, mrz_reader=None)
        with _client(store=_BrokenStore(), engines=engines) as client:
            response = client.get(f"{_API}/health")
            assert response.status_code == 503
            body = response.json()
            assert body["status"] == "degraded"
            assert body["redis"] is False
            assert body["models"] == {
                "face": False,
                "verify": False,
                "attributes": False,
                "antispoof": False,
                "ocr": False,
                "mrz": False,
                "structured": False,
            }

    def test_missing_engines_stay_200(self) -> None:
        engines = Engines(detector=None, ocr_reader=None, mrz_reader=None)
        with _client(store=MemoryJobStore(), engines=engines) as client:
            response = client.get(f"{_API}/health")
            assert response.status_code == 200
            assert response.json()["status"] == "ok"


class TestCrop:
    def _crop(self, client: TestClient, img: np.ndarray, side: str = "") -> Any:
        files = {"image": ("doc.png", _image_bytes(img), "image/png")}
        data = {"side": side} if side else None
        return client.post(f"{_API}/crop", files=files, data=data)

    def test_document_crop_passes(
        self, synthetic_document: Callable[..., np.ndarray]
    ) -> None:
        engines = _stub_engines()
        with _client(store=MemoryJobStore(), engines=engines) as client:
            response = self._crop(client, synthetic_document(), side="front")
            assert response.status_code == 200
            body = response.json()
            assert body["verdict"] == "pass"
            assert body["quad"] is not None and len(body["quad"]) == 4
            assert body["face_found"] is True
            assert body["checks"]["face_present"] == "pass"
            crop = _decode_base64_jpeg(body["crop_base64"])
            assert crop.shape[:2] == (body["height"], body["width"])

    def test_uniform_image_rejected_without_crop(self) -> None:
        with _client(store=MemoryJobStore()) as client:
            response = self._crop(client, np.full((600, 800, 3), 128, dtype=np.uint8))
            assert response.status_code == 200
            body = response.json()
            assert body["verdict"] == "reject"
            assert body["checks"]["document_found"] == "fail"
            assert body["quad"] is None
            assert body["crop_base64"] is None

    def test_back_side_face_not_expected(
        self, synthetic_document: Callable[..., np.ndarray]
    ) -> None:
        engines = _stub_engines(faces=[])
        with _client(store=MemoryJobStore(), engines=engines) as client:
            response = self._crop(client, synthetic_document(), side="back")
            assert response.status_code == 200
            body = response.json()
            assert body["checks"]["face_present"] == "n/a"
            assert body["verdict"] == "pass"

    def test_front_without_face_warns_not_rejects(
        self, synthetic_document: Callable[..., np.ndarray]
    ) -> None:
        engines = _stub_engines(faces=[])
        with _client(store=MemoryJobStore(), engines=engines) as client:
            response = self._crop(client, synthetic_document(), side="front")
            body = response.json()
            assert body["checks"]["face_present"] == "warn"
            assert body["verdict"] == "warn"

    def test_invalid_side_422(
        self, synthetic_document: Callable[..., np.ndarray]
    ) -> None:
        with _client(store=MemoryJobStore()) as client:
            response = self._crop(client, synthetic_document(), side="left")
            assert response.status_code == 422

    def test_oversized_413(
        self,
        monkeypatch: pytest.MonkeyPatch,
        synthetic_document: Callable[..., np.ndarray],
    ) -> None:
        with _env_client(
            monkeypatch,
            store=MemoryJobStore(),
            BLITZID_API_MAX_UPLOAD_MB="1",
        ) as client:
            response = self._crop(client, synthetic_document())
            assert response.status_code == 413

    def test_corrupt_bytes_400(self) -> None:
        with _client(store=MemoryJobStore()) as client:
            response = client.post(
                f"{_API}/crop", files={"image": ("d.png", b"garbage", "image/png")}
            )
            assert response.status_code == 400

    def test_pdf_bytes_415(self) -> None:
        with _client(store=MemoryJobStore()) as client:
            response = client.post(
                f"{_API}/crop",
                files={"image": ("d.pdf", b"%PDF-1.4 x", "application/pdf")},
            )
            assert response.status_code == 415

    def test_crop_face_check_holds_detector_lock(
        self, synthetic_document: Callable[..., np.ndarray]
    ) -> None:
        """The /crop face check must serialize via the engine lock."""

        class _RecordingDetector(_StubDetector):
            def __init__(self, lock: Any, faces: list[Any]) -> None:
                super().__init__(faces)
                self._lock = lock
                self.observed: list[bool] = []

            def detect_face_landmarks(self, image_input: Any) -> list[Any]:
                self.observed.append(self._lock.locked())
                return super().detect_face_landmarks(image_input)

        stub = _stub_engines()
        detector = _RecordingDetector(stub.detector_lock, [_face()])
        engines = Engines(
            detector=detector,
            ocr_reader=stub.ocr_reader,
            mrz_reader=stub.mrz_reader,
            detector_lock=stub.detector_lock,
        )
        with _client(store=MemoryJobStore(), engines=engines) as client:
            response = self._crop(client, synthetic_document(), side="front")
            assert response.status_code == 200
            assert detector.observed == [True]


class TestPrintedTexts:
    """The visual-zone filter must drop code-strip lines, keep everything else."""

    def _line(self, text: str) -> OCRText:
        return OCRText(bbox=(0, 0, 10, 10), text=text, confidence=0.9)

    def test_drops_mrz_lines(self) -> None:
        lines = [
            self._line("I<NLDSPECI20142999999990<<<<<8"),
            self._line("6503101F2403096NLD<<<<<<<<<<<8"),
            self._line("DE<BRUIJN<<WILLEKE<LISELOTTE<<"),
        ]
        assert _printed_texts(lines) == []

    def test_keeps_visual_zone_lines(self) -> None:
        lines = [
            self._line("SPECI2014"),
            self._line("999999990"),
            self._line("De Bruijn"),
            self._line("10 MAA/MAR 1965"),
            self._line("IDENTITY CARD"),
        ]
        assert _printed_texts(lines) == lines

    def test_keeps_short_alnum_runs(self) -> None:
        # 9 chars of digits is a printed field, not a 30-char code strip.
        assert [t.text for t in _printed_texts([self._line("999999990")])] == [
            "999999990"
        ]


class TestStructuredJob:
    def test_structured_section_returns_record(self) -> None:
        engines = _stub_engines()
        with _client(store=MemoryJobStore(), engines=engines) as client:
            response = _submit(client, types="structured")
            assert response.status_code == 202
            body = _poll_done(client, response.json()["job_id"])
            record = body["structured"]["record"]
            assert record["document_type"] == "id"
            assert record["fields"][0]["name"] == "surname"
            assert record["fields"][0]["value"] == "MUSTERMANN"
            assert record["raw_text"] == "ID"
            assert "processing_time_ms" in body["structured"]

    def test_structured_error_is_section_error(self) -> None:
        engines = _stub_engines(structured=ModelError("LLM endpoint unreachable"))
        with _client(store=MemoryJobStore(), engines=engines) as client:
            response = _submit(client, types="structured,face")
            body = _poll_done(client, response.json()["job_id"])
            assert body["structured"]["error"] == "LLM endpoint unreachable"
            assert "faces" in body["face"]

    def test_structured_engine_unavailable_400(self) -> None:
        engines = Engines(
            detector=None,
            ocr_reader=_StubOCRReader([_ocr_text()]),
            mrz_reader=None,
            structured_reader=None,
        )
        with _client(store=MemoryJobStore(), engines=engines) as client:
            assert _submit(client, types="structured").status_code == 400
            assert _submit(client, types="ocr").status_code == 202

    def test_structured_requires_ocr_reader_400(self) -> None:
        engines = Engines(
            detector=None,
            ocr_reader=None,
            mrz_reader=None,
            structured_reader=_StubStructuredReader(_structured_record()),
        )
        with _client(store=MemoryJobStore(), engines=engines) as client:
            assert _submit(client, types="structured").status_code == 400


class TestLiveness:
    def test_challenge_issues_random_action(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr("blitzid.api._liveness.secrets.randbelow", lambda n: 2)
        with _client(store=MemoryJobStore(), engines=_stub_engines()) as client:
            body = client.post(f"{_API}/liveness/challenge").json()
        assert body["action"] == "move_closer"
        assert body["expires_in"] == pytest.approx(120.0)

    def test_session_verifies_the_action(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr("blitzid.api._liveness.secrets.randbelow", lambda n: 2)
        detector = _SequenceDetector(
            [[_zoom_face(60)], [_zoom_face(70)], [_zoom_face(80)]]
        )
        stub = _stub_engines()
        engines = Engines(
            detector=detector,
            ocr_reader=stub.ocr_reader,
            mrz_reader=stub.mrz_reader,
            detector_lock=stub.detector_lock,
        )
        with _client(store=MemoryJobStore(), engines=engines) as client:
            challenge = client.post(f"{_API}/liveness/challenge").json()
            body = client.post(
                f"{_API}/liveness/session",
                data={"challenge_id": challenge["challenge_id"]},
                files=_frames(),
            ).json()
        assert body["live"] is True
        assert body["action"] == "move_closer"
        assert body["size_ratio"] == pytest.approx(8 / 6, rel=1e-3)

    def test_challenge_is_claim_once(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr("blitzid.api._liveness.secrets.randbelow", lambda n: 0)
        stub = _stub_engines()
        detector = _SequenceDetector([[_face()], [_face()], [_face()]])
        engines = Engines(
            detector=detector,
            ocr_reader=stub.ocr_reader,
            mrz_reader=stub.mrz_reader,
            detector_lock=stub.detector_lock,
        )
        with _client(store=MemoryJobStore(), engines=engines) as client:
            challenge = client.post(f"{_API}/liveness/challenge").json()
            data = {"challenge_id": challenge["challenge_id"]}
            first = client.post(f"{_API}/liveness/session", data=data, files=_frames())
            replay = client.post(f"{_API}/liveness/session", data=data, files=_frames())
        assert first.status_code == 200
        assert replay.status_code == 410

    def test_missing_face_in_a_frame_422(self) -> None:
        detector = _SequenceDetector([[_face()], [], [_face()]])
        stub = _stub_engines()
        engines = Engines(
            detector=detector,
            ocr_reader=stub.ocr_reader,
            mrz_reader=stub.mrz_reader,
            detector_lock=stub.detector_lock,
        )
        with _client(store=MemoryJobStore(), engines=engines) as client:
            challenge = client.post(f"{_API}/liveness/challenge").json()
            response = client.post(
                f"{_API}/liveness/session",
                data={"challenge_id": challenge["challenge_id"]},
                files=_frames(),
            )
        assert response.status_code == 422
        assert "frame 2" in response.json()["detail"]

    def test_liveness_engine_unavailable_400(self) -> None:
        engines = Engines(detector=None, ocr_reader=None, mrz_reader=None)
        with _client(store=MemoryJobStore(), engines=engines) as client:
            challenge = client.post(f"{_API}/liveness/challenge").json()
            response = client.post(
                f"{_API}/liveness/session",
                data={"challenge_id": challenge["challenge_id"]},
                files=_frames(),
            )
        assert response.status_code == 400


class TestAntiSpoofJob:
    def test_antispoof_section_reports_scores(self) -> None:
        engines = _stub_engines()
        with _client(store=MemoryJobStore(), engines=engines) as client:
            response = _submit(client, types="antispoof")
            assert response.status_code == 202
            body = _poll_done(client, response.json()["job_id"])["antispoof"]
        first = body["faces"][0]
        assert "processing_time_ms" in body
        assert first["bbox"] == [10, 10, 60, 60]
        assert first["live_score"] == pytest.approx(0.93)
        assert first["paper_score"] == pytest.approx(0.05)
        assert first["screen_score"] == pytest.approx(0.02)

    def test_antispoof_error_is_section_error(self) -> None:
        engines = _stub_engines(antispoof=ModelError("weights not loaded"))
        with _client(store=MemoryJobStore(), engines=engines) as client:
            response = _submit(client, types="antispoof,face")
            body = _poll_done(client, response.json()["job_id"])
            assert body["antispoof"]["error"] == "weights not loaded"
            assert "faces" in body["face"]

    def test_antispoof_engine_unavailable_400(self) -> None:
        engines = Engines(
            detector=None,
            ocr_reader=None,
            mrz_reader=None,
            antispoof_reader=None,
        )
        with _client(store=MemoryJobStore(), engines=engines) as client:
            assert _submit(client, types="antispoof").status_code == 400


class TestAttributesJob:
    def test_attributes_section_reports_faces(self) -> None:
        engines = _stub_engines()
        with _client(store=MemoryJobStore(), engines=engines) as client:
            response = _submit(client, types="attributes")
            assert response.status_code == 202
            body = _poll_done(client, response.json()["job_id"])["attributes"]
        assert "processing_time_ms" in body
        first = body["faces"][0]
        assert first["bbox"] == [10, 10, 60, 60]
        assert first["age"] == "30-39"
        assert first["gender"] == "Male"
        assert first["race"] == "White"

    def test_attributes_error_is_section_error(self) -> None:
        engines = _stub_engines(attributes=ModelError("weights not loaded"))
        with _client(store=MemoryJobStore(), engines=engines) as client:
            response = _submit(client, types="attributes,face")
            body = _poll_done(client, response.json()["job_id"])
            assert body["attributes"]["error"] == "weights not loaded"
            assert "faces" in body["face"]

    def test_attributes_engine_unavailable_400(self) -> None:
        engines = Engines(
            detector=None,
            ocr_reader=None,
            mrz_reader=None,
            attribute_reader=None,
        )
        with _client(store=MemoryJobStore(), engines=engines) as client:
            assert _submit(client, types="attributes").status_code == 400


class TestConsistencyJob:
    def test_consistency_section_reports_rows(self) -> None:
        engines = _stub_engines(structured=_consistent_record())
        with _client(store=MemoryJobStore(), engines=engines) as client:
            response = _submit(client, types="consistency")
            assert response.status_code == 202
            body = _poll_done(client, response.json()["job_id"])["consistency"]
        assert body["consistent"] is True
        verdicts = {row["field"]: row["verdict"] for row in body["comparisons"]}
        assert verdicts["document_number"] == "match"
        assert verdicts["date_of_birth"] == "match"
        assert verdicts["surname"] == "match"

    def test_consistency_rows_carry_both_sides(self) -> None:
        engines = _stub_engines(structured=_consistent_record())
        with _client(store=MemoryJobStore(), engines=engines) as client:
            response = _submit(client, types="consistency")
            body = _poll_done(client, response.json()["job_id"])["consistency"]
        birth = next(
            row for row in body["comparisons"] if row["field"] == "date_of_birth"
        )
        assert birth["mrz_value"] == "940120"
        assert birth["printed_value"] == "1994-01-20"
        assert birth["printed_confidence"] == pytest.approx(0.9)

    def test_photo_rows_check_age_and_sex(self) -> None:
        engines = _stub_engines(structured=_consistent_record())
        with _client(store=MemoryJobStore(), engines=engines) as client:
            response = _submit(client, types="consistency")
            body = _poll_done(client, response.json()["job_id"])["consistency"]
        photo = {row["field"]: row for row in body["photo_comparisons"]}
        assert photo["age"]["verdict"] == "match"
        assert photo["age"]["document_value"] == "32"
        assert photo["sex"]["verdict"] == "match"
        assert photo["sex"]["document_value"] == "M"

    def test_sections_share_per_job_work(self) -> None:
        engines = _stub_engines(structured=_consistent_record())
        with _client(store=MemoryJobStore(), engines=engines) as client:
            response = _submit(
                client, types="face,attributes,antispoof,ocr,mrz,structured,consistency"
            )
            body = _poll_done(client, response.json()["job_id"])
        assert body["status"] == "done"
        assert engines.detector.calls == 1
        assert engines.ocr_reader.calls == 1
        assert engines.mrz_reader.calls == 1
        assert engines.structured_reader.calls == 1
        assert engines.attribute_reader.calls == 1
        assert engines.antispoof_reader.calls == 1

    def test_mismatch_flips_consistent(self) -> None:
        record = StructuredOCR(
            document_type="id",
            fields=[
                ExtractedField(name="document_number", value="999", confidence=0.9)
            ],
            raw_text="ID",
        )
        engines = _stub_engines(structured=record)
        with _client(store=MemoryJobStore(), engines=engines) as client:
            response = _submit(client, types="consistency")
            body = _poll_done(client, response.json()["job_id"])["consistency"]
            assert body["consistent"] is False
            verdicts = {row["field"]: row["verdict"] for row in body["comparisons"]}
            assert verdicts["document_number"] == "mismatch"
            assert verdicts["surname"] == "unavailable"

    def test_consistency_error_is_section_error(self) -> None:
        engines = _stub_engines(structured=ModelError("LLM endpoint unreachable"))
        with _client(store=MemoryJobStore(), engines=engines) as client:
            response = _submit(client, types="consistency,face")
            body = _poll_done(client, response.json()["job_id"])
            assert body["consistency"]["error"] == "LLM endpoint unreachable"
            assert "faces" in body["face"]

    def test_consistency_engine_unavailable_400(self) -> None:
        engines = Engines(
            detector=None,
            ocr_reader=_StubOCRReader([_ocr_text()]),
            mrz_reader=None,
            structured_reader=_StubStructuredReader(_structured_record()),
        )
        with _client(store=MemoryJobStore(), engines=engines) as client:
            assert _submit(client, types="consistency").status_code == 400


class TestVerify:
    def _verify(
        self,
        client: TestClient,
        *,
        image_data: bytes | None = None,
        **data: str,
    ) -> Any:
        payload = image_data or _tiny_image_bytes()
        return client.post(
            f"{_API}/verify",
            files={
                "image1": ("a.png", payload, "image/png"),
                "image2": ("b.png", payload, "image/png"),
            },
            data=data,
        )

    def test_verify_returns_result(self) -> None:
        engines = _stub_engines()
        with _client(store=MemoryJobStore(), engines=engines) as client:
            response = self._verify(client)
            assert response.status_code == 200
            body = response.json()
            assert body["verified"] is True
            assert body["similarity"] == pytest.approx(0.97)
            assert body["threshold"] == pytest.approx(0.4)
            assert body["processing_time_ms"] == pytest.approx(123.0)
            assert body["backend"] == "ONNXRuntime"

    def test_verify_reports_face_evidence(self) -> None:
        engines = _stub_engines()
        with _client(store=MemoryJobStore(), engines=engines) as client:
            body = self._verify(client).json()
        assert body["face1"]["bbox"] == [10, 10, 60, 60]
        assert body["face1"]["confidence"] == pytest.approx(0.9)
        assert body["face1"]["alternatives"] == []
        assert body["face2"]["bbox"] == [10, 10, 60, 60]
        assert _decode_base64_jpeg(body["face1"]["crop_base64"]).ndim == 3

    def test_threshold_override_redecides_both_ways(self) -> None:
        strict = _stub_engines(
            verification=VerificationResult(
                verified=True,
                similarity=0.5,
                threshold=0.4,
                processing_time=0.01,
                backend="ONNXRuntime",
            )
        )
        with _client(store=MemoryJobStore(), engines=strict) as client:
            body = self._verify(client, threshold="0.9").json()
            assert body["verified"] is False
            assert body["threshold"] == pytest.approx(0.9)
        lenient = _stub_engines(
            verification=VerificationResult(
                verified=False,
                similarity=0.3,
                threshold=0.4,
                processing_time=0.01,
                backend="ONNXRuntime",
            )
        )
        with _client(store=MemoryJobStore(), engines=lenient) as client:
            body = self._verify(client, threshold="0.2").json()
            assert body["verified"] is True
            assert body["threshold"] == pytest.approx(0.2)

    def test_no_face_422(self) -> None:
        engines = _stub_engines(faces=[])
        with _client(store=MemoryJobStore(), engines=engines) as client:
            response = self._verify(client)
            assert response.status_code == 422
            assert "No face detected" in response.json()["detail"]

    def test_embedding_failure_422(self) -> None:
        engines = _stub_engines(verification=FaceVerificationError("alignment failed"))
        with _client(store=MemoryJobStore(), engines=engines) as client:
            assert self._verify(client).status_code == 422

    def test_pinned_bbox_selects_overlapping_face(self) -> None:
        engines = _stub_engines(faces=[_face(), _ghost_face()])
        big = _image_bytes(np.zeros((300, 300, 3), dtype=np.uint8))
        with _client(store=MemoryJobStore(), engines=engines) as client:
            response = self._verify(client, face1_bbox="200,200,30,30", image_data=big)
            assert response.status_code == 200
            body = response.json()
            assert body["face1"]["bbox"] == [200, 200, 30, 30]
            assert body["face1"]["alternatives"][0]["bbox"] == [10, 10, 60, 60]
        assert engines.verifier.compared[0][0].bbox == (200, 200, 30, 30)

    def test_unpinned_uses_highest_confidence_and_lists_alternatives(self) -> None:
        engines = _stub_engines(faces=[_ghost_face(), _face()])
        with _client(store=MemoryJobStore(), engines=engines) as client:
            body = self._verify(client).json()
            assert body["face1"]["bbox"] == [10, 10, 60, 60]
            assert [a["bbox"] for a in body["face1"]["alternatives"]] == [
                [200, 200, 30, 30]
            ]

    def test_pinned_bbox_without_face_422(self) -> None:
        engines = _stub_engines(faces=[_face()])
        with _client(store=MemoryJobStore(), engines=engines) as client:
            response = self._verify(client, face1_bbox="500,500,10,10")
            assert response.status_code == 422
            assert "No face" in response.json()["detail"]

    def test_malformed_bbox_422(self) -> None:
        with _client(store=MemoryJobStore(), engines=_stub_engines()) as client:
            assert self._verify(client, face1_bbox="1,2,3").status_code == 422
            assert self._verify(client, face2_bbox="a,b,c,d").status_code == 422
            assert self._verify(client, face1_bbox="10,10,0,5").status_code == 422

    def test_invalid_threshold_422(self) -> None:
        with _client(store=MemoryJobStore(), engines=_stub_engines()) as client:
            assert self._verify(client, threshold="2").status_code == 422

    def test_engine_unavailable_400(self) -> None:
        engines = Engines(detector=None, ocr_reader=None, mrz_reader=None)
        with _client(store=MemoryJobStore(), engines=engines) as client:
            assert self._verify(client).status_code == 400

    def test_missing_image_422(self) -> None:
        with _client(store=MemoryJobStore(), engines=_stub_engines()) as client:
            response = client.post(
                f"{_API}/verify",
                files={"image1": ("a.png", _tiny_image_bytes(), "image/png")},
            )
            assert response.status_code == 422

    def test_corrupt_bytes_400(self) -> None:
        with _client(store=MemoryJobStore(), engines=_stub_engines()) as client:
            response = client.post(
                f"{_API}/verify",
                files={
                    "image1": ("a.png", _tiny_image_bytes(), "image/png"),
                    "image2": ("b.png", b"garbage", "image/png"),
                },
            )
            assert response.status_code == 400

    def test_pdf_bytes_415(self) -> None:
        with _client(store=MemoryJobStore(), engines=_stub_engines()) as client:
            response = client.post(
                f"{_API}/verify",
                files={
                    "image1": ("a.pdf", b"%PDF-1.4 x", "application/pdf"),
                    "image2": ("b.png", _tiny_image_bytes(), "image/png"),
                },
            )
            assert response.status_code == 415

    def test_oversized_413(
        self,
        monkeypatch: pytest.MonkeyPatch,
        synthetic_document: Callable[..., np.ndarray],
    ) -> None:
        with _env_client(
            monkeypatch,
            store=MemoryJobStore(),
            engines=_stub_engines(),
            BLITZID_API_MAX_UPLOAD_MB="1",
        ) as client:
            response = client.post(
                f"{_API}/verify",
                files={
                    "image1": (
                        "a.png",
                        _image_bytes(synthetic_document()),
                        "image/png",
                    ),
                    "image2": ("b.png", _tiny_image_bytes(), "image/png"),
                },
            )
            assert response.status_code == 413

    def test_verify_holds_detector_lock(self) -> None:
        """/verify must serialize detector access via the engine lock."""

        class _RecordingDetector(_StubDetector):
            def __init__(self, lock: Any, faces: list[Any]) -> None:
                super().__init__(faces)
                self._lock = lock
                self.observed: list[bool] = []

            def detect_face_landmarks(self, image_input: Any) -> list[Any]:
                self.observed.append(self._lock.locked())
                return super().detect_face_landmarks(image_input)

        stub = _stub_engines()
        detector = _RecordingDetector(stub.detector_lock, [_face()])
        engines = Engines(
            detector=detector,
            ocr_reader=stub.ocr_reader,
            mrz_reader=stub.mrz_reader,
            verifier=stub.verifier,
            structured_reader=stub.structured_reader,
            detector_lock=stub.detector_lock,
        )
        with _client(store=MemoryJobStore(), engines=engines) as client:
            response = self._verify(client)
            assert response.status_code == 200
            assert detector.observed == [True, True]


class TestRedisJobStore:
    def _store(self, **kwargs: int) -> RedisJobStore:
        return RedisJobStore(
            fakeredis.FakeStrictRedis(),
            ttl_seconds=kwargs.get("ttl_seconds", 900),
            max_queued=kwargs.get("max_queued", 50),
            lease_seconds=kwargs.get("lease_seconds", 120),
        )

    def _submit(self, store: RedisJobStore, job_id: str) -> None:
        store.submit(job_id, b"image-bytes", ["face"])

    def test_submit_claim_fifo(self) -> None:
        store = self._store()
        self._submit(store, "a")
        self._submit(store, "b")
        first = store.claim()
        assert first is not None and first.job_id == "a"
        assert first.image == b"image-bytes"
        assert first.types == ["face"]
        second = store.claim()
        assert second is not None and second.job_id == "b"
        assert store.claim() is None

    def test_status_transitions(self) -> None:
        store = self._store()
        self._submit(store, "a")
        assert store.get("a").status == "queued"
        claimed = store.claim()
        assert claimed is not None and claimed.job_id == "a"
        assert store.get("a").status == "running"
        store.write_result("a", claimed.token, {"state": "done"})
        snapshot = store.get("a")
        assert snapshot.status == "done"
        assert snapshot.result == {"state": "done"}
        assert store.get("a").status == "gone"

    def test_unknown_id_missing(self) -> None:
        assert self._store().get("nope").status == "missing"

    def test_max_queued_raises(self) -> None:
        store = self._store(max_queued=1)
        self._submit(store, "a")
        with pytest.raises(QueueFullError, match="queue is full"):
            self._submit(store, "b")

    def test_concurrent_submits_respect_cap(self) -> None:
        store = self._store(max_queued=5)
        accepted: list[str] = []
        rejected: list[str] = []
        barrier = Barrier(8)

        def worker(index: int) -> None:
            barrier.wait()
            try:
                self._submit(store, f"job-{index}")
            except QueueFullError:
                rejected.append(f"job-{index}")
            else:
                accepted.append(f"job-{index}")

        threads = [Thread(target=worker, args=(i,)) for i in range(8)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()
        assert len(accepted) == 5
        assert len(rejected) == 3
        assert int(store._redis.llen(QUEUE_KEY)) == 5

    def test_claim_skips_expired_job(self) -> None:
        store = self._store()
        self._submit(store, "expired")
        self._submit(store, "fresh")
        store._redis.delete(JOB_PREFIX + "expired")
        claimed = store.claim()
        assert claimed is not None and claimed.job_id == "fresh"
        assert store.claim() is None
        # the expired id is retired: polls see gone, not forever-queued
        assert store.get("expired").status == "gone"

    def test_renew_extends_claim_and_status_lease(self) -> None:
        store = self._store(lease_seconds=120)
        self._submit(store, "a")
        claimed = store.claim()
        assert claimed is not None
        assert store.renew("a", claimed.token) is True
        assert 0 < int(store._redis.ttl(CLAIM_PREFIX + "a")) <= 120
        store._redis.delete(CLAIM_PREFIX + "a")
        assert store.renew("a", claimed.token) is False

    def test_renew_rejects_foreign_token(self) -> None:
        store = self._store(lease_seconds=120)
        self._submit(store, "a")
        claimed = store.claim()
        assert claimed is not None
        assert store.renew("a", "not-the-owner") is False
        assert store.renew("a", claimed.token) is True

    def test_write_result_rejects_stale_claim(self) -> None:
        store = self._store(lease_seconds=120)
        self._submit(store, "a")
        claimed = store.claim()
        assert claimed is not None
        store._redis.set(CLAIM_PREFIX + "a", "other-token", ex=120)
        with pytest.raises(StaleClaimError):
            store.write_result("a", claimed.token, {"state": "done"})
        assert store.get("a").status == "running"

    def test_write_result_frees_job_payload(self) -> None:
        store = self._store()
        self._submit(store, "a")
        claimed = store.claim()
        assert claimed is not None
        store.write_result("a", claimed.token, {"state": "done"})
        assert not store._redis.exists(JOB_PREFIX + "a")
        assert not store._redis.exists(CLAIM_PREFIX + "a")

    def test_results_index_self_prunes_on_write(self) -> None:
        store = self._store()
        self._submit(store, "a")
        claimed = store.claim()
        assert claimed is not None
        store.write_result("a", claimed.token, {"state": "done"})
        assert store.counts()["stored"] == 1
        # simulate the result key's TTL window passing without a
        # claim-once read: the index entry's expiry score is due
        store._redis.zadd("blitzid:results", {"a": 0})
        self._submit(store, "b")
        claimed_b = store.claim()
        assert claimed_b is not None
        store.write_result("b", claimed_b.token, {"state": "done"})
        assert store.counts()["stored"] == 1

    def test_requeued_id_is_claimed_once(self) -> None:
        """When the sweeper requeues an id that a worker popped but
        did not claim, the next claim turn has exactly one winner."""
        store = self._store()
        self._submit(store, "a")
        # worker pops the id; sweeper requeues it before any claim
        store._redis.lmove(QUEUE_KEY, "blitzid:processing", "RIGHT", "LEFT")
        store._redis.lpush(QUEUE_KEY, "a")
        first = store.claim()
        assert first is not None and first.job_id == "a"
        # a second claimant racing the same queue turn loses to the
        # token owner (the claim key exists and holds the token)
        assert store.renew("a", "imposter-token") is False
        with pytest.raises(StaleClaimError):
            store.write_result("a", "imposter-token", {"state": "done"})
        assert store.get("a").status == "running"

    def test_sweep_requeues_expired_leases_only(self) -> None:
        store = self._store()
        self._submit(store, "dead")
        self._submit(store, "live")
        store.claim()
        store.claim()
        store._redis.delete(CLAIM_PREFIX + "dead")
        assert store.sweep() == 1
        assert store.get("dead").status == "queued"
        assert store.get("live").status == "running"

    def test_sweep_respects_queue_cap_and_drops_expired(self) -> None:
        store = self._store(max_queued=2)
        self._submit(store, "queued")
        self._submit(store, "dead")
        # both claimed, then both workers die; the queue refills to cap
        claimed_queued = store.claim()
        claimed_dead = store.claim()
        assert claimed_queued is not None and claimed_dead is not None
        self._submit(store, "filler1")
        self._submit(store, "filler2")
        store._redis.delete(CLAIM_PREFIX + "queued", CLAIM_PREFIX + "dead")
        # queue at cap: the live-payload id stays in processing
        assert store.sweep() == 0
        assert store.get("queued").status == "running"
        assert int(store._redis.llen(QUEUE_KEY)) == 2
        # expired payload: dropped from processing instead of requeued
        store._redis.delete(JOB_PREFIX + "dead")
        assert store.sweep() == 1
        # the id stays known (gone, not missing) — the client that
        # submitted it learns its result will never come
        assert store.get("dead").status == "gone"
        assert int(store._redis.llen(PROCESSING_KEY)) == 1

    def test_counts_and_ping(self) -> None:
        store = self._store()
        self._submit(store, "a")
        claimed = store.claim()
        assert claimed is not None
        store.write_result("a", claimed.token, {"state": "done"})
        self._submit(store, "b")
        assert store.ping() is True
        assert store.counts() == {"queued": 1, "running": 0, "stored": 1}
        assert store._redis.llen(QUEUE_KEY) == 1
        assert store.get("a").status == "done"  # claim-once drops stored count
        assert store.counts()["stored"] == 0


_INTEGRATION = bool(os.environ.get("BLITZID_OCR_INTEGRATION"))


class TestRealEngines:
    """Real-engine lifecycle, gated behind BLITZID_OCR_INTEGRATION=1."""

    @pytest.mark.skipif(
        not _INTEGRATION, reason="set BLITZID_OCR_INTEGRATION=1 to download models"
    )
    def test_specimen_full_pipeline(self, monkeypatch: pytest.MonkeyPatch) -> None:
        pytest.importorskip("rapidocr")
        fixtures = Path(__file__).parent.parent / "images"
        specimen = fixtures / "nl_td1_id_specimen.jpg"
        ocr_reader = RapidOCRReader(log_level=logging.WARNING)
        engines = Engines(
            detector=FaceDetectorDNN(enable_cache=False, log_level=logging.WARNING),
            ocr_reader=ocr_reader,
            mrz_reader=MRZReader(reader=ocr_reader, log_level=logging.WARNING),
        )
        with _env_client(
            monkeypatch, store=MemoryJobStore(), engines=engines
        ) as client:
            data = _image_bytes(cv2.imread(str(specimen)), ".jpg")
            response = _submit(client, types="face,ocr,mrz", data=data)
            assert response.status_code == 202
            result = _poll_done(client, response.json()["job_id"], timeout=60.0)
            assert result["status"] == "done"
            assert len(result["face"]["faces"]) == 1
            assert result["face"]["faces"][0]["confidence"] > 0.5
            assert result["ocr"]["lines"]
            record = result["mrz"]["record"]
            assert record["mrz_type"] == "TD1"
            assert record["surname"] == "DE BRUIJN"
            assert record["given_names"] == "WILLEKE LISELOTTE"
