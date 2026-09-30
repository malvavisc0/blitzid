"""Analysis dispatch for /analyze jobs: decode, run, build sections."""

from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import asdict, dataclass, field
from threading import Lock
from typing import Any

import numpy as np

from .._image import MIN_DIMENSION
from ..exceptions import BlitzIDError, ModelError
from ..face._face import Face
from ..face.detector import FaceDetectorDNN
from ..reading.mrz import MRZReader
from ..reading.ocr import RapidOCRReader
from ._upload import decode_image_bytes, encode_jpeg_b64

FACE_CROP_PADDING = 0.2


@dataclass
class Engines:
    """Shared analysis engine singletons with one lock per engine.

    The locks serialize access to engines that are not thread-safe:
    the cache-less detector and the RapidOCR pipeline shared by the
    OCR and MRZ sections. Fields are None when an engine is
    unavailable (missing ``ocr`` extra, missing weights).
    """

    detector: FaceDetectorDNN | None
    ocr_reader: RapidOCRReader | None
    mrz_reader: MRZReader | None
    detector_lock: Lock = field(default_factory=Lock)
    ocr_lock: Lock = field(default_factory=Lock)


def run_job(image: bytes, types: list[str], engines: Engines) -> dict[str, Any]:
    """Run the requested analyses and build the result dict.

    Per-section failures (``BlitzIDError`` subclasses) are captured as
    section errors; other sections still return.

    Returns:
        ``{"state": "done", "<type>": <section>, ...}`` — duplicates
        deduped, one section per requested type.
    """
    img = decode_image_bytes(image)
    if img is None or min(img.shape[:2]) < MIN_DIMENSION:
        return {"state": "failed", "error": "stored image bytes are undecodable"}
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


def _crop_face(img: np.ndarray, bbox: tuple[int, int, int, int]) -> np.ndarray:
    """Crop a face region with relative padding, clipped to the image."""
    height, width = img.shape[:2]
    x, y, face_w, face_h = bbox
    pad_w, pad_h = int(face_w * FACE_CROP_PADDING), int(face_h * FACE_CROP_PADDING)
    x1, y1 = max(0, x - pad_w), max(0, y - pad_h)
    x2 = min(width, x + face_w + pad_w)
    y2 = min(height, y + face_h + pad_h)
    return img[y1:y2, x1:x2]


def _face_item(img: np.ndarray, face: Face) -> dict[str, Any]:
    """Build one face result item (bbox, confidence, landmarks, crop)."""
    crop = _crop_face(img, face.bbox)
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


_SECTION_RUNNERS: dict[str, Callable[[np.ndarray, Engines], dict[str, Any]]] = {
    "face": _run_face,
    "ocr": _run_ocr,
    "mrz": _run_mrz,
}
