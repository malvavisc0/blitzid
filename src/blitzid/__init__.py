"""BlitzID — DNN-based face detection.

Public API
----------
- ``FaceDetectorDNN`` — the primary OpenCV DNN detector.
- ``FaceDetectorDeepFace`` — optional DeepFace-based backend.
- ``DetectionMetrics`` — metrics dataclass returned by ``detect_face_with_metrics``.
- ``BlitzIDError``, ``ModelError``, ``ImageError`` — exception hierarchy.
"""

from .deepface import FaceDetectorDeepFace
from .detector import DetectionMetrics, FaceDetectorDNN
from .exceptions import BlitzIDError, ImageError, ModelError

# Backward-compat aliases for the old exception names.
FaceDetectorError = BlitzIDError
ModelDownloadError = ModelError
ImageLoadError = ImageError
ImageProcessingError = ImageError
CUDAConfigError = ModelError
InvalidParameterError = BlitzIDError
OptionalDependencyError = BlitzIDError

__all__ = [
    "BlitzIDError",
    "CUDAConfigError",
    "DetectionMetrics",
    "FaceDetectorDNN",
    "FaceDetectorDeepFace",
    "FaceDetectorError",
    "ImageError",
    "ImageLoadError",
    "ImageProcessingError",
    "InvalidParameterError",
    "ModelDownloadError",
    "ModelError",
    "OptionalDependencyError",
]
