"""Analysis dispatch for /analyze jobs: decode, run, build sections."""

from __future__ import annotations

import time
from collections.abc import Callable, Mapping
from dataclasses import asdict, dataclass, field
from threading import Lock
from types import MappingProxyType
from typing import Any

import numpy as np

from .. import (
    Face,
    FaceDetectorDNN,
    FaceVerifier,
    MRZReader,
    RapidOCRReader,
    StructuredOCRReader,
)
from .._image import crop_with_padding
from ..exceptions import BlitzIDError, ModelError
from ._upload import decode_image_bytes, dims_within_bounds, encode_jpeg_b64

FACE_CROP_PADDING = 0.2


@dataclass
class Engines:
    """Shared analysis engine singletons with one lock per engine.

    The locks serialize access to engines that are not thread-safe:
    the cache-less detector, the RapidOCR pipeline shared by the OCR,
    MRZ, and structured sections, and the LLM reader's agent cache.
    Fields are None when an engine is unavailable (missing ``ocr``
    extra, missing weights, missing LLM configuration).
    """

    detector: FaceDetectorDNN | None
    ocr_reader: RapidOCRReader | None
    mrz_reader: MRZReader | None
    verifier: FaceVerifier | None = None
    structured_reader: StructuredOCRReader | None = None
    detector_lock: Lock = field(default_factory=Lock)
    ocr_lock: Lock = field(default_factory=Lock)
    structured_lock: Lock = field(default_factory=Lock)


def run_job(image: bytes, types: list[str], engines: Engines) -> dict[str, Any]:
    """Run the requested analyses and build the result dict.

    Per-section failures (``BlitzIDError`` subclasses) are captured as
    section errors; other sections still return.

    Returns:
        ``{"state": "done", "<type>": <section>, ...}`` — duplicates
        deduped, one section per requested type.
    """
    img = decode_image_bytes(image)
    if img is None or not dims_within_bounds(img):
        return {"state": "failed", "error": "stored image bytes failed validation"}
    sections: dict[str, Any] = {"state": "done"}
    for analysis_type in dict.fromkeys(types):
        sections[analysis_type] = _run_section(analysis_type, img, engines)
    return sections


def _run_section(
    analysis_type: str, img: np.ndarray, engines: Engines
) -> dict[str, Any]:
    """Run one analysis, capturing BlitzIDError as a section error."""
    start = time.perf_counter()
    try:
        body = _SECTION_RUNNERS[analysis_type](img, engines)
    except BlitzIDError as e:
        body = {"error": str(e)}
    body["processing_time_ms"] = round((time.perf_counter() - start) * 1000, 1)
    return body


def _face_item(img: np.ndarray, face: Face) -> dict[str, Any]:
    """Build one face result item (bbox, confidence, landmarks, crop)."""
    crop = crop_with_padding(img, face.bbox, FACE_CROP_PADDING)
    return {
        "bbox": list(face.bbox),
        "confidence": face.confidence,
        "landmarks": [[x, y] for x, y in face.landmarks],
        "crop_base64": encode_jpeg_b64(crop),
    }


def _run_face(img: np.ndarray, engines: Engines) -> dict[str, Any]:
    """Detect faces and build the face section."""
    detector = engines.detector
    if detector is None:
        raise ModelError("face engine unavailable (SCRFD weights not loaded)")
    with engines.detector_lock:
        faces = detector.detect_face_landmarks(img)
    return {"faces": [_face_item(img, face) for face in faces]}


def _run_ocr(img: np.ndarray, engines: Engines) -> dict[str, Any]:
    """Read text lines and build the ocr section."""
    reader = engines.ocr_reader
    if reader is None:
        raise ModelError("ocr analysis requires the blitzid[ocr] extra")
    with engines.ocr_lock:
        texts = reader.read(img)
    return {
        "lines": [
            {
                "bbox": list(text.bbox),
                "text": text.text,
                "confidence": text.confidence,
            }
            for text in texts
        ]
    }


def _run_mrz(img: np.ndarray, engines: Engines) -> dict[str, Any]:
    """Read the MRZ and build the mrz section (shares the OCR engine)."""
    reader = engines.mrz_reader
    if reader is None:
        raise ModelError("mrz analysis requires the blitzid[ocr] extra")
    with engines.ocr_lock:
        record = reader.read(img)
    return {"record": asdict(record)}


def _run_structured(img: np.ndarray, engines: Engines) -> dict[str, Any]:
    """Read text lines, then rebuild a typed record via the LLM."""
    reader = engines.structured_reader
    ocr_reader = engines.ocr_reader
    if reader is None or ocr_reader is None:
        raise ModelError(
            "structured analysis requires the blitzid[ocr] extra "
            "and LLM configuration (BLITZID_LLM_*)"
        )
    with engines.ocr_lock:
        texts = ocr_reader.read(img)
    with engines.structured_lock:
        record = reader.read(texts)
    return {"record": record.model_dump(mode="json")}


_SECTION_RUNNERS: Mapping[str, Callable[[np.ndarray, Engines], dict[str, Any]]] = (
    MappingProxyType(
        {
            "face": _run_face,
            "ocr": _run_ocr,
            "mrz": _run_mrz,
            "structured": _run_structured,
        }
    )
)
