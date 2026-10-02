"""HTTP API smoke test — real engines and endpoints, run in-process.

Boots the FastAPI app with real weights (plus the configured LLM
endpoint for the ``structured`` job) and walks the identity flow over
the committed document fixtures: ``/health``, face counts per document
type (passport, driver licenses, ID card), ``/verify`` with and without
face pinning (evidence, alternatives, threshold), ``structured`` and
``consistency`` jobs, and one full passport job covering picture, OCR,
MRZ, structured fields, and the code-strip cross-check together. This
is a smoke test: it loads every engine and makes network calls, so it
is run on demand, not as part of the pytest suite. It uses an
in-process Redis fake, so no Redis server is needed.

Run example::

    uv run --env-file .env python scripts/api_smoke.py

Exits non-zero if any endpoint misbehaves. The BLITZID_LLM_* trio must
be set (the structured engine is checked in ``/health``).
"""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Any, cast

import cv2
import fakeredis
import redis
from fastapi.testclient import TestClient

from blitzid.api._app import create_app
from blitzid.api._jobs import RedisJobStore

_IMAGES = Path("images")
_PASSPORT = _IMAGES / "td3_passport_specimen.jpg"
_ID_CARD = _IMAGES / "bub_der_personalausweis_kopie.jpg"
_NL_SPECIMEN = _IMAGES / "nl_td1_id_specimen.jpg"
_LICENSES = (
    _IMAGES / "nys_driver_license_sample.webp",
    _IMAGES / "tlc_driver_license_sample.png",
)
_CONFERENCE = _IMAGES / "solvay_conference_1927.jpg"

_JOB_TIMEOUT = 180.0
_POLL_INTERVAL = 0.2


def _upload(path: Path) -> tuple[str, bytes, str]:
    return (path.name, path.read_bytes(), "application/octet-stream")


def _poll(client: TestClient, job_id: str) -> dict[str, Any]:
    deadline = time.monotonic() + _JOB_TIMEOUT
    while time.monotonic() < deadline:
        response = client.get(f"/jobs/{job_id}")
        assert response.status_code == 200, response.text
        body: dict[str, Any] = response.json()
        if body["status"] in ("done", "failed"):
            assert body["status"] == "done", body
            return body
        time.sleep(_POLL_INTERVAL)
    raise AssertionError(f"job {job_id} did not finish within {_JOB_TIMEOUT:.0f}s")


def _submit_job(client: TestClient, path: Path, types: str) -> dict[str, Any]:
    response = client.post(
        "/analyze", files={"image": _upload(path)}, data={"types": types}
    )
    assert response.status_code == 202, response.text
    return _poll(client, response.json()["job_id"])


def _section(body: dict[str, Any], name: str) -> dict[str, Any]:
    """Return one job section, failing with its error when it holds one."""
    section: dict[str, Any] = body[name]
    assert "error" not in section, f"{name} section failed: {section['error']}"
    return section


def _check_health(client: TestClient) -> None:
    print("— /health —")
    body = client.get("/health").json()
    print(f"  models: {body['models']}")
    assert body["redis"] is True, body
    assert all(body["models"].values()), f"engine(s) missing: {body['models']}"


def _check_face_counts(client: TestClient) -> None:
    """Report how many portraits each document type presents.

    The count decides how careful callers must be: one portrait means
    auto-selection is safe, two (a main photo plus a faint ghost
    portrait) means the caller should pin the face it wants.
    """
    print("— faces per document type —")
    for path in (_PASSPORT, *_LICENSES, _ID_CARD):
        faces = _submit_job(client, path, "face")["face"]["faces"]
        scores = [round(face["confidence"], 2) for face in faces]
        print(f"  {path.name}: {len(faces)} face(s), confidences={scores}")
        if path in (_PASSPORT, _ID_CARD):
            assert faces, f"{path.name}: expected at least one portrait"


def _check_verify_auto(client: TestClient) -> None:
    print("— /verify (passport portrait vs itself) —")
    response = client.post(
        "/verify", files={"image1": _upload(_PASSPORT), "image2": _upload(_PASSPORT)}
    )
    assert response.status_code == 200, response.text
    body = response.json()
    print(f"  verified={body['verified']} similarity={body['similarity']:.3f}")
    print(
        f"  face1 bbox={body['face1']['bbox']}, "
        f"alternatives={len(body['face1']['alternatives'])}"
    )
    assert body["verified"] is True, body
    assert body["similarity"] > 0.9, body
    assert body["face1"]["crop_base64"], "evidence crop missing"
    assert body["face2"]["crop_base64"], "evidence crop missing"


def _check_verify_pinned(client: TestClient) -> None:
    print("— /verify (two people from the conference photo, pinned) —")
    faces = _submit_job(client, _CONFERENCE, "face")["face"]["faces"]
    assert len(faces) >= 2, f"need two faces for the pinning check: {len(faces)}"
    first, second = faces[0]["bbox"], faces[1]["bbox"]
    pinned = {
        "face1_bbox": ",".join(str(v) for v in first),
        "face2_bbox": ",".join(str(v) for v in second),
    }
    files = {"image1": _upload(_CONFERENCE), "image2": _upload(_CONFERENCE)}
    body = client.post("/verify", files=files, data=pinned).json()
    print(
        f"  compared face1={body['face1']['bbox']} vs face2={body['face2']['bbox']}, "
        f"similarity={body['similarity']:.3f} verified={body['verified']}"
    )
    assert body["face1"]["bbox"] == first and body["face2"]["bbox"] == second, body
    assert len(body["face1"]["alternatives"]) == len(faces) - 1, body["face1"]
    assert body["verified"] is False, body
    relaxed = client.post(
        "/verify", files=files, data={**pinned, "threshold": "-1"}
    ).json()
    print(f"  same pair at threshold=-1: verified={relaxed['verified']}")
    assert relaxed["verified"] is True, relaxed
    assert relaxed["threshold"] == -1.0, relaxed


def _check_structured_job(client: TestClient) -> None:
    print("— structured extraction job (ID card) —")
    record = _submit_job(client, _ID_CARD, "structured")["structured"]["record"]
    print(f"  document_type={record['document_type']}, fields={len(record['fields'])}")
    for field in record["fields"][:5]:
        print(f"    {field['name']}: {field['value']} (conf={field['confidence']:.2f})")
    assert record["document_type"] == "id", record
    assert record["fields"], "no fields extracted"
    assert record["raw_text"], "raw_text should carry the OCR lines"


def _check_consistency_job(client: TestClient) -> None:
    print("— consistency job (TD1 specimen) —")
    section = _submit_job(client, _NL_SPECIMEN, "consistency")["consistency"]
    for row in section["comparisons"]:
        print(
            f"  {row['field']}: {row['verdict']} "
            f"(mrz={row['mrz_value']!r} printed={row['printed_value']!r})"
        )
    for row in section["photo_comparisons"]:
        print(
            f"  photo {row['field']}: {row['verdict']} "
            f"(doc={row['document_value']!r} photo={row['photo_value']!r})"
        )
    verdicts = {row["field"]: row["verdict"] for row in section["comparisons"]}
    strict = ["document_number", "date_of_birth", "date_of_expiry", "sex"]
    assert all(verdicts[field] == "match" for field in strict), verdicts
    assert verdicts["surname"] in ("match", "partial"), verdicts
    assert verdicts["given_names"] in ("match", "partial"), verdicts


def _check_attributes_job(client: TestClient) -> None:
    print("— attributes job (ID card portrait) —")
    faces = _submit_job(client, _ID_CARD, "attributes")["attributes"]["faces"]
    assert faces, "no face found"
    first = faces[0]
    print(f"  age={first['age']} gender={first['gender']} race={first['race']}")
    assert first["age"] and first["gender"] and first["race"], first


def _check_antispoof_job(client: TestClient) -> None:
    print("— antispoof job (ID card portrait) —")
    faces = _submit_job(client, _ID_CARD, "antispoof")["antispoof"]["faces"]
    assert faces, "no face found"
    first = faces[0]
    print(
        f"  live={first['live_score']} paper={first['paper_score']} "
        f"screen={first['screen_score']}"
    )
    total = first["live_score"] + first["paper_score"] + first["screen_score"]
    assert 0.99 <= total <= 1.01, first


def _check_liveness(client: TestClient) -> None:
    print("— liveness (zoom frames vs a random challenge) —")
    img = cv2.imread(str(_ID_CARD))
    assert img is not None, f"unreadable fixture: {_ID_CARD}"
    zooms = [cv2.resize(img, None, fx=scale, fy=scale) for scale in (1.0, 1.2, 1.5)]
    files = {
        name: (f"{name}.png", cv2.imencode(".png", frame)[1].tobytes(), "image/png")
        for name, frame in zip(("frame1", "frame2", "frame3"), zooms, strict=True)
    }
    challenge = client.post("/liveness/challenge").json()
    data = {"challenge_id": challenge["challenge_id"]}
    body = client.post("/liveness/session", data=data, files=files).json()
    print(
        f"  action={challenge['action']} live={body['live']} "
        f"size_ratio={body['size_ratio']}"
    )
    if challenge["action"] == "move_closer":
        assert body["live"] is True, body
    else:
        assert body["live"] is False, body
    replay = client.post("/liveness/session", data=data, files=files)
    assert replay.status_code == 410, replay.text


def _check_passport_scenario(client: TestClient) -> None:
    """One job over the whole passport: picture, printed text, code strip.

    The serial assert is deliberately soft: the printed serial is the
    noisiest extraction (a small glyph run through the LLM), so the
    smoke only requires it to have been compared.
    """
    print("— passport scenario (picture + OCR + MRZ + structured + consistency) —")
    body = _submit_job(client, _PASSPORT, "face,ocr,mrz,structured,consistency")
    faces = _section(body, "face")["faces"]
    lines = _section(body, "ocr")["lines"]
    record = _section(body, "mrz")["record"]
    printed = _section(body, "structured")["record"]
    report = _section(body, "consistency")
    print(f"  picture: {len(faces)} portrait(s)")
    print(f"  printed text: {len(lines)} OCR line(s)")
    print(
        f"  code strip: {record['mrz_type']} {record['document_number']} "
        f"birth={record['birth_date']} expiry={record['expiry_date']}"
    )
    fields = printed["fields"]
    print(f"  structured: {printed['document_type']}, {len(fields)} field(s)")
    verdicts = {row["field"]: row["verdict"] for row in report["comparisons"]}
    photo = {row["field"]: row["verdict"] for row in report["photo_comparisons"]}
    print(f"  consistency: {verdicts} consistent={report['consistent']}")
    print(f"  photo: {photo}")
    assert faces, "passport portrait not found"
    assert lines, "no printed text read"
    assert record["mrz_type"] == "TD3", record
    assert record["document_number"], record
    assert printed["fields"], "no printed fields extracted"
    assert verdicts["document_number"] != "unavailable", verdicts
    assert verdicts["date_of_birth"] != "unavailable", verdicts


def main() -> None:
    """Run the smoke test and exit non-zero on failure."""
    logging.basicConfig(level=logging.WARNING)
    store = RedisJobStore(
        cast("redis.Redis", fakeredis.FakeStrictRedis()),
        ttl_seconds=900,
        max_queued=50,
        lease_seconds=120,
    )
    with TestClient(create_app(store=store)) as client:
        _check_health(client)
        _check_face_counts(client)
        _check_verify_auto(client)
        _check_verify_pinned(client)
        _check_structured_job(client)
        _check_consistency_job(client)
        _check_attributes_job(client)
        _check_antispoof_job(client)
        _check_liveness(client)
        _check_passport_scenario(client)
    print("Smoke test passed.")


if __name__ == "__main__":
    main()
