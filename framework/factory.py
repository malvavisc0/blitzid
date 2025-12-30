"""Factory methods for creating pre-configured detectors."""

import logging
from pathlib import Path
from typing import Optional

from .deepface_facade import FaceDetectorDeepFace
from .facade import FaceDetectorDNN

# Keep default model resolution consistent with the facade.
DEFAULT_MODEL_DIR = Path(__file__).resolve().parents[1] / "models"


class FaceDetectorFactory:
    """Factory for creating pre-configured detectors."""

    @staticmethod
    def create_fast_detector(
        model_dir: Optional[Path] = None,
        log_level: int = logging.WARNING,
        require_cuda: bool = False,
    ) -> "FaceDetectorDNN":
        """
        Create speed-optimized detector.

        Args:
            model_dir: Directory for model files (uses default if None)
            log_level: Logging level

        Returns:
            FaceDetectorDNN configured for fast detection
        """

        if model_dir is None:
            model_dir = DEFAULT_MODEL_DIR

        return FaceDetectorDNN(
            confidence_threshold=0.7,  # Higher threshold = fewer detections
            min_face_size=(80, 80),  # Larger minimum = faster
            nms_threshold=0.4,  # More aggressive NMS
            log_level=log_level,
            enable_cache=True,
            max_cache_size=200,
            model_dir=model_dir,
            require_cuda=require_cuda,
        )

    @staticmethod
    def create_accurate_detector(
        model_dir: Optional[Path] = None,
        log_level: int = logging.INFO,
        require_cuda: bool = False,
    ) -> "FaceDetectorDNN":
        """
        Create accuracy-optimized detector.

        Args:
            model_dir: Directory for model files (uses default if None)
            log_level: Logging level

        Returns:
            FaceDetectorDNN configured for accurate detection
        """

        if model_dir is None:
            model_dir = DEFAULT_MODEL_DIR

        return FaceDetectorDNN(
            confidence_threshold=0.3,  # Lower threshold = more detections
            min_face_size=(20, 20),  # Smaller minimum = detect small faces
            nms_threshold=0.2,  # Less aggressive NMS
            log_level=log_level,
            enable_cache=False,
            model_dir=model_dir,
            require_cuda=require_cuda,
        )

    @staticmethod
    def create_balanced_detector(
        model_dir: Optional[Path] = None,
        log_level: int = logging.INFO,
        require_cuda: bool = False,
    ) -> "FaceDetectorDNN":
        """
        Create balanced detector.

        Args:
            model_dir: Directory for model files (uses default if None)
            log_level: Logging level

        Returns:
            FaceDetectorDNN configured for balanced performance
        """

        if model_dir is None:
            model_dir = DEFAULT_MODEL_DIR

        return FaceDetectorDNN(
            confidence_threshold=0.5,
            min_face_size=(50, 50),
            nms_threshold=0.3,
            log_level=log_level,
            enable_cache=True,
            max_cache_size=100,
            model_dir=model_dir,
            require_cuda=require_cuda,
        )

    @staticmethod
    def create_deepface_detector(
        model_dir: Optional[Path] = None,
        log_level: int = logging.INFO,
        detector_backend: str = "retinaface",
        align: bool = True,
        enforce_detection: bool = False,
    ) -> "FaceDetectorDeepFace":
        """Create a DeepFace-based detector+aligner.

        This backend is slower but provides high-quality detection and alignment.
        It stores DeepFace weights under `models/deepface/` by default.

        Args:
            model_dir: Directory for DeepFace weights (default: `models/deepface/`)
            log_level: Logging level
            detector_backend: DeepFace detector backend (recommended: "retinaface")
            align: Whether to return aligned crops
            enforce_detection: If True, DeepFace raises when no face is found

        Returns:
            A [`framework.deepface_facade.FaceDetectorDeepFace`](framework/deepface_facade.py:1)
            instance.
        """

        # Keep DeepFace weights separate from OpenCV DNN model files.
        if model_dir is None:
            model_dir = DEFAULT_MODEL_DIR / "deepface"

        return FaceDetectorDeepFace(
            model_dir=model_dir,
            detector_backend=detector_backend,
            align=align,
            enforce_detection=enforce_detection,
            log_level=log_level,
        )
