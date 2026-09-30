"""BlitzID — DNN-based face detection.

Public API
----------
- ``FaceDetectorDNN`` — the SCRFD-based face detector.
- ``Face`` — detection record (bbox, confidence, landmarks).
- ``DetectionMetrics`` — metrics dataclass returned by ``detect_face_with_metrics``.
- ``BlitzIDError``, ``ModelError``, ``ImageError`` — exception hierarchy.
"""

from .exceptions import BlitzIDError, ImageError, ModelError
from .face._face import DetectionMetrics, Face
from .face.detector import FaceDetectorDNN

FaceDetectorError = BlitzIDError
ModelDownloadError = ModelError
ImageLoadError = ImageError
ImageProcessingError = ImageError
InvalidParameterError = BlitzIDError

__all__ = [
    "BlitzIDError",
    "DetectionMetrics",
    "Face",
    "FaceDetectorDNN",
    "FaceDetectorError",
    "ImageError",
    "ImageLoadError",
    "ImageProcessingError",
    "InvalidParameterError",
    "ModelDownloadError",
    "ModelError",
]
