"""BlitzID — DNN-based face detection.

Public API
----------
- ``FaceDetectorDNN`` — the SCRFD-based face detector.
- ``Face`` — detection record (bbox, confidence, landmarks).
- ``DetectionMetrics`` — metrics dataclass returned by ``detect_face_with_metrics``.
- ``RapidOCRReader`` — the RapidOCR-based text reader (``ocr`` extra).
- ``OCRText`` — recognized text line (bbox, text, confidence).
- ``MRZReader`` — the ICAO 9303 machine-readable zone reader (``ocr`` extra).
- ``MRZRecord`` — parsed MRZ fields.
- ``DocumentCropper`` — document localization, perspective crop, and QC.
- ``QualityReport`` — document crop quality-check verdict.
- ``BlitzIDError``, ``ModelError``, ``ImageError``, ``MRZError`` — exception hierarchy.
"""

from .exceptions import BlitzIDError, ImageError, ModelError, MRZError
from .face._face import DetectionMetrics, Face
from .face.detector import FaceDetectorDNN
from .reading.document import DocumentCropper, QualityReport
from .reading.mrz import MRZReader, MRZRecord
from .reading.ocr import OCRText, RapidOCRReader

FaceDetectorError = BlitzIDError
ModelDownloadError = ModelError
ImageLoadError = ImageError
ImageProcessingError = ImageError
InvalidParameterError = BlitzIDError

__all__ = [
    "BlitzIDError",
    "DetectionMetrics",
    "DocumentCropper",
    "Face",
    "FaceDetectorDNN",
    "FaceDetectorError",
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
]
