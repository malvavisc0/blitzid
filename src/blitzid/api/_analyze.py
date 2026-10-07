"""Analysis dispatch for /analyze jobs: decode, run, build sections.

Sections of one job share a small memo, so detection, OCR, and the LLM
each run at most once per image no matter how many types are requested:
``attributes`` and ``antispoof`` classify the ``face`` section's
detections, and ``mrz`` parses the ``ocr`` section's text lines.
"""

from __future__ import annotations

import re
import time
from collections.abc import Callable, Mapping
from dataclasses import asdict, dataclass, field
from threading import Lock
from types import MappingProxyType
from typing import Any, cast

import numpy as np

from blitzid._image import crop_with_padding
from blitzid.api._upload import decode_image_bytes, dims_within_bounds, encode_jpeg_b64
from blitzid.exceptions import BlitzIDError, ModelError
from blitzid.face._antispoof import AntiSpoofReader
from blitzid.face._attributes import FaceAttributeReader, FaceAttributes
from blitzid.face._face import Face
from blitzid.face.detector import FaceDetectorDNN
from blitzid.face.verifier import FaceVerifier
from blitzid.reading.barcode import BarcodeReader, _record_dict
from blitzid.reading.consistency import cross_check
from blitzid.reading.mrz import MRZReader, MRZRecord
from blitzid.reading.ocr import OCRText, RapidOCRReader
from blitzid.reading.structurize import StructuredOCR, StructuredOCRReader

FACE_CROP_PADDING = 0.2

# A machine-readable code-strip line: 15+ characters of only A-Z, 0-9
# and '<' with no spaces (TD1 3x30, TD2 2x36, TD3 2x44). Visual-zone
# lines carry words, separators, or shorter values, so they never match.
_MRZ_LINE = re.compile(r"[A-Za-z0-9<]{15,}")


@dataclass
class Engines:
    """Shared analysis engine singletons with one lock per engine.

    The locks serialize access to the engines that are not thread-safe:
    ``detector_lock`` guards the one SCRFD ``FaceDetectorDNN`` instance
    (every detection, including those behind the attribute, anti-spoof,
    verification, and document-crop paths), ``ocr_lock`` the RapidOCR
    pipeline (its one text pass per job feeds the OCR, MRZ, and
    structured sections), and ``structured_lock`` the LLM reader. The
    FairFace, MiniFASNet, and ArcFace passes are plain onnxruntime
    sessions, which are thread-safe, so they run outside any lock, and
    the zxing-cpp barcode decode is a stateless per-call function.
    Fields are None when an engine is unavailable (missing ``ocr`` or
    ``barcode`` extra, missing weights, missing LLM configuration).
    """

    detector: FaceDetectorDNN | None
    ocr_reader: RapidOCRReader | None
    mrz_reader: MRZReader | None
    barcode_reader: BarcodeReader | None = None
    verifier: FaceVerifier | None = None
    structured_reader: StructuredOCRReader | None = None
    attribute_reader: FaceAttributeReader | None = None
    antispoof_reader: AntiSpoofReader | None = None
    detector_lock: Lock = field(default_factory=Lock)
    ocr_lock: Lock = field(default_factory=Lock)
    structured_lock: Lock = field(default_factory=Lock)


def run_job(image: bytes, types: list[str], engines: Engines) -> dict[str, Any]:
    """Run the requested analyses and build the result dict.

    Per-section failures (``BlitzIDError`` subclasses) are captured as
    section errors; other sections still return. A per-job memo shares
    detection, OCR, and LLM results between sections that need them.

    Returns:
        ``{"state": "done", "<type>": <section>, ...}`` — duplicates
        deduped, one section per requested type.
    """
    img = decode_image_bytes(image)
    if img is None or not dims_within_bounds(img):
        return {"state": "failed", "error": "stored image bytes failed validation"}
    sections: dict[str, Any] = {"state": "done"}
    memo: dict[str, Any] = {}
    for analysis_type in dict.fromkeys(types):
        sections[analysis_type] = _run_section(analysis_type, img, engines, memo)
    return sections


def _run_section(
    analysis_type: str, img: np.ndarray, engines: Engines, memo: dict[str, Any]
) -> dict[str, Any]:
    """Run one analysis, capturing BlitzIDError as a section error."""
    start = time.perf_counter()
    try:
        body = _SECTION_RUNNERS[analysis_type](img, engines, memo)
    except BlitzIDError as e:
        body = {"error": str(e)}
    body["processing_time_ms"] = round((time.perf_counter() - start) * 1000, 1)
    return body


def _memoized[T](memo: dict[str, Any], key: str, load: Callable[[], T]) -> T:
    """Compute one shared per-job value at most once."""
    if key not in memo:
        memo[key] = load()
    return cast(T, memo[key])


def _face_item(img: np.ndarray, face: Face) -> dict[str, Any]:
    """Build one face result item (bbox, confidence, landmarks, crop)."""
    crop = crop_with_padding(img, face.bbox, FACE_CROP_PADDING)
    return {
        "bbox": list(face.bbox),
        "confidence": face.confidence,
        "landmarks": [[x, y] for x, y in face.landmarks],
        "crop_base64": encode_jpeg_b64(crop),
    }


def _read_faces(img: np.ndarray, engines: Engines, memo: dict[str, Any]) -> list[Face]:
    """Detect the image's faces (shared between sections)."""
    detector = engines.detector
    if detector is None:
        raise ModelError("face engine unavailable (SCRFD weights not loaded)")

    def _load() -> list[Face]:
        with engines.detector_lock:
            return detector.detect_face_landmarks(img)

    return _memoized(memo, "faces", _load)


def _read_texts(
    img: np.ndarray, engines: Engines, memo: dict[str, Any]
) -> list[OCRText]:
    """Read the image's text lines (shared between sections)."""
    reader = engines.ocr_reader
    if reader is None:
        raise ModelError("ocr analysis requires the blitzid[ocr] extra")

    def _load() -> list[OCRText]:
        with engines.ocr_lock:
            return reader.read(img)

    return _memoized(memo, "texts", _load)


def _read_attributes(
    img: np.ndarray, engines: Engines, memo: dict[str, Any]
) -> list[FaceAttributes]:
    """Predict attributes for the image's faces (shared between sections)."""
    reader = engines.attribute_reader
    if reader is None:
        raise ModelError(
            "attributes analysis requires the SCRFD detector and FairFace weights"
        )

    def _load() -> list[FaceAttributes]:
        return reader.read_faces(img, _read_faces(img, engines, memo))

    return _memoized(memo, "attributes", _load)


def _read_mrz(img: np.ndarray, engines: Engines, memo: dict[str, Any]) -> MRZRecord:
    """Parse the MRZ from the shared OCR text lines."""
    reader = engines.mrz_reader
    if reader is None:
        raise ModelError("mrz analysis requires the blitzid[ocr] extra")

    def _load() -> MRZRecord:
        return reader.parse(_read_texts(img, engines, memo))

    return _memoized(memo, "mrz", _load)


def _read_structured(
    img: np.ndarray, engines: Engines, memo: dict[str, Any]
) -> StructuredOCR:
    """Read text lines, then rebuild a typed record via the LLM."""
    reader = engines.structured_reader
    if reader is None or engines.ocr_reader is None:
        raise ModelError(
            "structured analysis requires the blitzid[ocr] extra "
            "and LLM configuration (BLITZID_LLM_*)"
        )

    def _load() -> StructuredOCR:
        texts = _printed_texts(_read_texts(img, engines, memo))
        with engines.structured_lock:
            return reader.read(texts)

    return _memoized(memo, "structured", _load)


def _printed_texts(texts: list[OCRText]) -> list[OCRText]:
    """Keep the visual zone's lines, dropping machine-readable code-strip lines.

    The code strip (lines of A-Z, 0-9 and '<') is parsed deterministically
    by the ``mrz`` reader; fed to the visual-zone LLM extraction it makes
    the model glue code-strip values into printed fields (e.g. appending
    the MRZ's personal number to ``document_number``).
    """
    return [text for text in texts if not _MRZ_LINE.fullmatch(text.text)]


def _run_face(
    img: np.ndarray, engines: Engines, memo: dict[str, Any]
) -> dict[str, Any]:
    """Build the face section."""
    faces = _read_faces(img, engines, memo)
    return {"faces": [_face_item(img, face) for face in faces]}


def _run_attributes(
    img: np.ndarray, engines: Engines, memo: dict[str, Any]
) -> dict[str, Any]:
    """Build the attributes section (age group, gender, race per face)."""
    attributes = _read_attributes(img, engines, memo)
    return {"faces": [asdict(item) for item in attributes]}


def _run_antispoof(
    img: np.ndarray, engines: Engines, memo: dict[str, Any]
) -> dict[str, Any]:
    """Build the antispoof section (live/paper/screen scores per face)."""
    reader = engines.antispoof_reader
    if reader is None:
        raise ModelError(
            "antispoof analysis requires the SCRFD detector and MiniFASNet weights"
        )
    scores = reader.read_faces(img, _read_faces(img, engines, memo))
    return {"faces": [asdict(item) for item in scores]}


def _run_ocr(img: np.ndarray, engines: Engines, memo: dict[str, Any]) -> dict[str, Any]:
    """Build the ocr section."""
    texts = _read_texts(img, engines, memo)
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


def _run_mrz(img: np.ndarray, engines: Engines, memo: dict[str, Any]) -> dict[str, Any]:
    """Build the mrz section."""
    return {"record": asdict(_read_mrz(img, engines, memo))}


def _run_barcode(
    img: np.ndarray, engines: Engines, memo: dict[str, Any]
) -> dict[str, Any]:
    """Build the barcode section."""
    reader = engines.barcode_reader
    if reader is None:
        raise ModelError("barcode analysis requires the blitzid[barcode] extra")
    return {"record": _record_dict(reader.read(img))}


def _run_structured(
    img: np.ndarray, engines: Engines, memo: dict[str, Any]
) -> dict[str, Any]:
    """Build the structured section."""
    return {"record": _read_structured(img, engines, memo).model_dump(mode="json")}


def _run_consistency(
    img: np.ndarray, engines: Engines, memo: dict[str, Any]
) -> dict[str, Any]:
    """Cross-check the printed fields and photo against the document data."""
    if (
        engines.mrz_reader is None
        or engines.structured_reader is None
        or engines.ocr_reader is None
    ):
        raise ModelError(
            "consistency analysis requires the blitzid[ocr] extra "
            "and LLM configuration (BLITZID_LLM_*)"
        )
    photo = None
    if engines.attribute_reader is not None:
        attributes = _read_attributes(img, engines, memo)
        if attributes:
            photo = max(attributes, key=lambda item: item.confidence)
    report = cross_check(
        _read_structured(img, engines, memo),
        _read_mrz(img, engines, memo),
        photo=photo,
    )
    return asdict(report)


_SECTION_RUNNERS: Mapping[
    str, Callable[[np.ndarray, Engines, dict[str, Any]], dict[str, Any]]
] = MappingProxyType(
    {
        "face": _run_face,
        "attributes": _run_attributes,
        "antispoof": _run_antispoof,
        "ocr": _run_ocr,
        "mrz": _run_mrz,
        "barcode": _run_barcode,
        "structured": _run_structured,
        "consistency": _run_consistency,
    }
)
