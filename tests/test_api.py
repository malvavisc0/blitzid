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
    Face,
    FaceDetectorDNN,
    MRZError,
    MRZReader,
    MRZRecord,
    OCRText,
    RapidOCRReader,
)

pytest.importorskip("fastapi")

import fakeredis
import redis
from fastapi.testclient import TestClient

from blitzid.api._analyze import Engines
from blitzid.api._app import ApiConfig, _JobCancelled, _Workers, create_app
from blitzid.api._jobs import (
    CLAIM_PREFIX,
    JOB_PREFIX,
    PROCESSING_KEY,
    QUEUE_KEY,
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
    token-owned with expiry timestamps (wound forward via
    :meth:`expire_all_claims`), results are claim-once, and the
    submission registry prunes to a 2x-TTL grace window so expired ids
    become ``missing`` (404), not ``gone`` (410), after the window.
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

    def expire_all_claims(self) -> None:
        """Simulate every worker dying: drop all claim leases."""
        self._claims.clear()

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
            if job is not None:
                break
        else:
            return None
        token = uuid.uuid4().hex
        if job_id in self._claims:
            # requeued by the sweeper mid-claim: leave it queued
            self._processing.remove(job_id)
            return None
        self._processing.append(job_id)
        self._status[job_id] = "running"
        self._claims[job_id] = (token, _now() + self._lease)
        return ClaimedJob(job_id=job_id, image=job[0], types=job[1], token=token)

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

    def detect_face_landmarks(self, image_input: Any) -> list[Any]:
        if isinstance(self._faces, Exception):
            raise self._faces
        return self._faces


class _StubOCRReader:
    """RapidOCRReader stand-in returning a fixed result or error."""

    def __init__(self, texts: list[OCRText] | Exception) -> None:
        self._texts = texts

    def read(self, image_input: Any) -> list[OCRText]:
        if isinstance(self._texts, Exception):
            raise self._texts
        return self._texts


class _StubMRZReader:
    """MRZReader stand-in returning a fixed record or error."""

    def __init__(self, record: MRZRecord | Exception) -> None:
        self._record = record

    def read(self, image_input: Any) -> MRZRecord:
        if isinstance(self._record, Exception):
            raise self._record
        return self._record


def _face() -> Face:
    return Face(
        bbox=(10, 10, 60, 60),
        confidence=0.9,
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


def _stub_engines(
    faces: list[Any] | Exception | None = None,
    ocr_texts: list[OCRText] | Exception | None = None,
    mrz: MRZRecord | Exception | None = None,
) -> Engines:
    return Engines(
        detector=_StubDetector([_face()] if faces is None else faces),
        ocr_reader=_StubOCRReader([_ocr_text()] if ocr_texts is None else ocr_texts),
        mrz_reader=_StubMRZReader(_mrz_record() if mrz is None else mrz),
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
        "/analyze",
        files={"image": ("doc.png", data or _tiny_image_bytes(), "image/png")},
        data={"types": types},
    )


def _poll_done(client: TestClient, job_id: str, timeout: float = 5.0) -> dict[str, Any]:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        response = client.get(f"/jobs/{job_id}")
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
            assert body["status_url"] == f"/jobs/{body['job_id']}"
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
            assert client.get(f"/jobs/{job_id}").status_code == 410

    def test_unknown_job_id_404(self) -> None:
        with _client(store=MemoryJobStore()) as client:
            response = client.get("/jobs/doesnotexist")
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
            assert client.get(f"/jobs/{job_id}").json()["status"] == "queued"
            time.sleep(1.3)
            response = client.get(f"/jobs/{job_id}")
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

    def test_execute_skips_result_write_when_cancelled(self) -> None:
        """_execute on a job whose claim was lost writes no result —
        the re-claimed worker owns the job."""

        class _NoWriteStore(MemoryJobStore):
            def write_result(
                self, job_id: str, token: str, result: dict[str, Any]
            ) -> None:
                raise AssertionError("cancelled job must not write a result")

        config = ApiConfig(
            redis_url="redis://localhost:6379/0",
            max_upload_bytes=20 * 1024 * 1024,
            ttl_seconds=900,
            max_queued=50,
            max_concurrent=1,
            lease_seconds=120,
        )
        store = _NoWriteStore()
        workers = _Workers(store, _stub_engines(), config)
        store.submit("job", _tiny_image_bytes(), ["face"])
        claimed = store.claim()
        assert claimed is not None

        # the heartbeat lost the claim mid-run and raised the cancel
        cancelled = Event()
        cancelled.set()
        with (
            pytest.raises(_JobCancelled),
            workers._lease_heartbeat(claimed, cancelled),
        ):
            pass


class TestSubmitValidation:
    def test_unknown_type_422(self) -> None:
        with _client(store=MemoryJobStore()) as client:
            assert _submit(client, types="lipsum").status_code == 422

    def test_missing_types_422(self) -> None:
        with _client(store=MemoryJobStore()) as client:
            response = client.post(
                "/analyze", files={"image": ("d.png", _tiny_image_bytes(), "image/png")}
            )
            assert response.status_code == 422

    def test_missing_image_422(self) -> None:
        with _client(store=MemoryJobStore()) as client:
            response = client.post("/analyze", data={"types": "face"})
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
            response = client.get("/health")
            assert response.status_code == 200
            body = response.json()
            assert body["status"] == "ok"
            assert body["models"] == {"face": True, "ocr": True, "mrz": True}
            assert body["redis"] is True
            assert set(body["jobs"]) == {"queued", "running", "stored"}

    def test_degraded_without_engines_and_redis(self) -> None:
        engines = Engines(detector=None, ocr_reader=None, mrz_reader=None)
        with _client(store=_BrokenStore(), engines=engines) as client:
            response = client.get("/health")
            assert response.status_code == 503
            body = response.json()
            assert body["status"] == "degraded"
            assert body["redis"] is False
            assert body["models"] == {"face": False, "ocr": False, "mrz": False}

    def test_missing_engines_stay_200(self) -> None:
        engines = Engines(detector=None, ocr_reader=None, mrz_reader=None)
        with _client(store=MemoryJobStore(), engines=engines) as client:
            response = client.get("/health")
            assert response.status_code == 200
            assert response.json()["status"] == "ok"


class TestCrop:
    def _crop(self, client: TestClient, img: np.ndarray, side: str = "") -> Any:
        files = {"image": ("doc.png", _image_bytes(img), "image/png")}
        data = {"side": side} if side else None
        return client.post("/crop", files=files, data=data)

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
                "/crop", files={"image": ("d.png", b"garbage", "image/png")}
            )
            assert response.status_code == 400

    def test_pdf_bytes_415(self) -> None:
        with _client(store=MemoryJobStore()) as client:
            response = client.post(
                "/crop", files={"image": ("d.pdf", b"%PDF-1.4 x", "application/pdf")}
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
