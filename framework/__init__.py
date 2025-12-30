"""Face Detector Module - Modular DNN-based face detection.

This package provides a clean, modular architecture for face detection using
OpenCV's DNN module with optional CUDA acceleration.

Public API notes
----------------
- `FaceDetectorDNN` is implemented in [`framework/facade.py`](framework/facade.py:1) to
  avoid circular imports between the package initializer and the factory.
- The package continues to support `from framework import FaceDetectorDNN`.
"""

from .deepface_facade import FaceDetectorDeepFace
from .exceptions import (
    CUDAConfigError,
    FaceDetectorError,
    ImageLoadError,
    ImageProcessingError,
    InvalidParameterError,
    ModelDownloadError,
    OptionalDependencyError,
)

# Re-export the public facade API
from .facade import FaceDetectorDNN
from .factory import FaceDetectorFactory
from .metrics import DetectionMetrics

# Public API exports
__all__ = [
    "FaceDetectorDNN",
    "FaceDetectorDeepFace",
    "FaceDetectorError",
    "ModelDownloadError",
    "ImageLoadError",
    "ImageProcessingError",
    "CUDAConfigError",
    "InvalidParameterError",
    "OptionalDependencyError",
    "DetectionMetrics",
    "FaceDetectorFactory",
]
