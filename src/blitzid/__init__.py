"""BlitzID — DNN-based face detection and verification.

Public API
----------
- ``FaceDetectorDNN`` — the SCRFD-based face detector.
- ``Face`` — detection record (bbox, confidence, landmarks).
- ``DetectionMetrics`` — metrics dataclass returned by ``detect_face_with_metrics``.
- ``FaceVerifier`` — the ArcFace-based 1:1 face verifier.
- ``VerificationResult`` — verification outcome (verdict, similarity, threshold).
- ``FaceAttributeReader`` — age group, gender, and race per face.
- ``FaceAttributes`` — one face's predicted attributes.
- ``AntiSpoofReader`` — passive live-vs-spoof scores per face.
- ``AntiSpoofResult`` — one face's spoof scores (paper, real, screen).
- ``RapidOCRReader`` — the RapidOCR-based text reader (``ocr`` extra).
- ``OCRText`` — recognized text line (bbox, text, confidence).
- ``MRZReader`` — the ICAO 9303 machine-readable zone reader (``ocr`` extra).
- ``MRZRecord`` — parsed MRZ fields.
- ``BarcodeReader`` — the AAMVA PDF417 driver's-license reader
  (``barcode`` extra).
- ``BarcodeRecord`` — parsed AAMVA payload fields.
- ``StructuredOCRReader`` — LLM-backed structured output from OCR lines
  (``ocr`` extra).
- ``StructuredOCR``, ``ExtractedField`` — structured-output schemas.
- ``cross_check`` — field-by-field comparison of the printed fields
  against the machine-readable zone and the portrait photo.
- ``ConsistencyReport``, ``FieldComparison``, ``PhotoComparison`` —
  consistency records.
- ``DocumentCropper`` — document localization, perspective crop, and QC.
- ``QualityReport`` — document crop quality-check verdict.
- ``BlitzIDError``, ``ModelError``, ``ImageError``, ``MRZError``,
  ``BarcodeError``, ``FaceVerificationError`` — exception hierarchy.
"""

from blitzid.exceptions import (
    BarcodeError,
    BlitzIDError,
    FaceVerificationError,
    ImageError,
    ModelError,
    MRZError,
)
from blitzid.face._antispoof import AntiSpoofReader, AntiSpoofResult
from blitzid.face._attributes import FaceAttributeReader, FaceAttributes
from blitzid.face._face import DetectionMetrics, Face, VerificationResult
from blitzid.face.detector import FaceDetectorDNN
from blitzid.face.verifier import FaceVerifier
from blitzid.reading.barcode import BarcodeReader, BarcodeRecord
from blitzid.reading.consistency import (
    ConsistencyReport,
    FieldComparison,
    PhotoComparison,
    cross_check,
)
from blitzid.reading.document import DocumentCropper, QualityReport
from blitzid.reading.mrz import MRZReader, MRZRecord
from blitzid.reading.ocr import OCRText, RapidOCRReader
from blitzid.reading.structurize import (
    ExtractedField,
    StructuredOCR,
    StructuredOCRReader,
)

FaceDetectorError = BlitzIDError
ModelDownloadError = ModelError
ImageLoadError = ImageError
ImageProcessingError = ImageError
InvalidParameterError = BlitzIDError

__all__ = [
    "AntiSpoofReader",
    "AntiSpoofResult",
    "BarcodeError",
    "BarcodeReader",
    "BarcodeRecord",
    "BlitzIDError",
    "ConsistencyReport",
    "DetectionMetrics",
    "DocumentCropper",
    "ExtractedField",
    "Face",
    "FaceAttributeReader",
    "FaceAttributes",
    "FaceDetectorDNN",
    "FaceDetectorError",
    "FaceVerificationError",
    "FaceVerifier",
    "FieldComparison",
    "ImageError",
    "ImageLoadError",
    "ImageProcessingError",
    "InvalidParameterError",
    "MRZError",
    "MRZReader",
    "MRZRecord",
    "ModelDownloadError",
    "ModelError",
    "OCRText",
    "PhotoComparison",
    "QualityReport",
    "RapidOCRReader",
    "StructuredOCR",
    "StructuredOCRReader",
    "VerificationResult",
    "cross_check",
]
