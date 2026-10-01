"""BlitzID — DNN-based face detection and verification.

Public API
----------
- ``FaceDetectorDNN`` — the SCRFD-based face detector.
- ``Face`` — detection record (bbox, confidence, landmarks).
- ``DetectionMetrics`` — metrics dataclass returned by ``detect_face_with_metrics``.
- ``FaceVerifier`` — the ArcFace-based 1:1 face verifier.
- ``VerificationResult`` — verification outcome (verdict, similarity, threshold).
- ``RapidOCRReader`` — the RapidOCR-based text reader (``ocr`` extra).
- ``OCRText`` — recognized text line (bbox, text, confidence).
- ``MRZReader`` — the ICAO 9303 machine-readable zone reader (``ocr`` extra).
- ``MRZRecord`` — parsed MRZ fields.
- ``StructuredOCRReader`` — LLM-backed structured output from OCR lines
  (``ocr`` extra).
- ``StructuredOCR``, ``ExtractedField`` — structured-output schemas.
- ``DocumentCropper`` — document localization, perspective crop, and QC.
- ``QualityReport`` — document crop quality-check verdict.
- ``BlitzIDError``, ``ModelError``, ``ImageError``, ``MRZError``,
  ``FaceVerificationError`` — exception hierarchy.
"""

from .exceptions import (
    BlitzIDError,
    FaceVerificationError,
    ImageError,
    ModelError,
    MRZError,
)
from .face._face import DetectionMetrics, Face, VerificationResult
from .face.detector import FaceDetectorDNN
from .face.verifier import FaceVerifier
from .reading.document import DocumentCropper, QualityReport
from .reading.mrz import MRZReader, MRZRecord
from .reading.ocr import OCRText, RapidOCRReader
from .reading.structurize import ExtractedField, StructuredOCR, StructuredOCRReader

FaceDetectorError = BlitzIDError
ModelDownloadError = ModelError
ImageLoadError = ImageError
ImageProcessingError = ImageError
InvalidParameterError = BlitzIDError

__all__ = [
    "BlitzIDError",
    "DetectionMetrics",
    "DocumentCropper",
    "ExtractedField",
    "Face",
    "FaceDetectorDNN",
    "FaceDetectorError",
    "FaceVerificationError",
    "FaceVerifier",
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
    "QualityReport",
    "RapidOCRReader",
    "StructuredOCR",
    "StructuredOCRReader",
    "VerificationResult",
]
