"""Facade API for the face detection system.

This module contains the high-level `FaceDetectorDNN` class (the public-facing API)
that composes the internal components (model manager, loader, processor, cache, etc.).

It intentionally does *not* import the factory at module import time to avoid
circular imports between the package initializer and the factory.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING, Dict, List, Optional, Tuple, Union

import numpy as np
from numpy.typing import NDArray

from .cache import DetectionCache
from .detector import FaceDetector
from .exceptions import FaceDetectorError, InvalidParameterError
from .image_loader import ImageLoader
from .metrics import DetectionMetrics, MetricsTracker
from .models import ModelManager
from .processors import FaceProcessor
from .visualizer import FaceVisualizer

if TYPE_CHECKING:  # pragma: no cover
    from PIL.Image import Image as PILImageType
else:
    PILImageType = object

ImageInput = Union[Path, str, NDArray[np.uint8], "PILImageType"]

# Default model directory
# Resolve relative to the repository/package root (not the current working directory)
# to avoid `Path.cwd()` surprises when the library is used from other projects.
BASE_DIR = Path(__file__).resolve().parents[1]
MODEL_DIR = BASE_DIR / "models"


class FaceDetectorDNN:
    """Facade for face detection system.

    Provides a simple, unified interface to the modular face detection system
    while maintaining backward compatibility with the original API.
    """

    def __init__(
        self,
        confidence_threshold: float = 0.5,
        min_face_size: Tuple[int, int] = (50, 50),
        nms_threshold: float = 0.3,
        log_level: int = logging.INFO,
        enable_cache: bool = False,
        max_cache_size: int = 100,
        model_dir: Optional[Path] = None,
        preprocess: bool = True,
        multi_scale: bool = False,
        scales: Tuple[float, ...] = (1.0, 1.5, 2.0),
        require_cuda: bool = False,
    ):
        """Initialize Face Detector.

        Args:
            confidence_threshold: Minimum confidence for face detection (0.0-1.0)
            min_face_size: Minimum face size as (width, height) in pixels
            nms_threshold: IoU threshold for Non-Maximum Suppression (0.0-1.0)
            log_level: Logging level (logging.DEBUG, INFO, WARNING, ERROR)
            enable_cache: Enable caching of detection results
            max_cache_size: Maximum number of cached results
            model_dir: Directory for model files (uses default if None)
            preprocess: If True, apply basic image cleanup (CLAHE on luminance)

        Raises:
            InvalidParameterError: If parameters are invalid
        """
        # Validate parameters
        self._validate_parameters(
            confidence_threshold, min_face_size, nms_threshold, max_cache_size
        )

        # Use default model directory if not specified
        if model_dir is None:
            model_dir = MODEL_DIR

        # Setup logger
        self.logger = self._setup_logger(log_level)

        # Create components
        self.model_manager = ModelManager(model_dir, self.logger)
        self.image_loader = ImageLoader(self.logger, preprocess=preprocess)
        self.processor = FaceProcessor(nms_threshold)
        self.cache = DetectionCache(max_cache_size, enable_cache, self.logger)

        # Create detector
        self.detector = FaceDetector(
            model_manager=self.model_manager,
            image_loader=self.image_loader,
            processor=self.processor,
            cache=self.cache,
            confidence_threshold=confidence_threshold,
            min_face_size=min_face_size,
            logger=self.logger,
            multi_scale=multi_scale,
            scales=scales,
            use_cuda=True,
            require_cuda=require_cuda,
        )

        # Create visualizer
        self.visualizer = FaceVisualizer(self.logger)

        # Create metrics tracker
        self.metrics = MetricsTracker()

        # Store configuration
        self.require_cuda = bool(require_cuda)
        self.confidence_threshold = confidence_threshold
        self.min_face_size = min_face_size
        self.nms_threshold = nms_threshold
        self.enable_cache = enable_cache
        self.max_cache_size = max_cache_size
        self.preprocess = preprocess
        self.multi_scale = multi_scale
        self.scales = scales
        self.backend_type = self.detector.backend
        self.net = self.detector.net

    @staticmethod
    def _validate_parameters(
        confidence_threshold: float,
        min_face_size: Tuple[int, int],
        nms_threshold: float,
        max_cache_size: int,
    ) -> None:
        """Validate constructor parameters.

        Raises:
            InvalidParameterError: If any parameter is invalid
        """
        if confidence_threshold < 0.0 or confidence_threshold > 1.0:
            raise InvalidParameterError(
                f"confidence_threshold must be between 0.0 and 1.0, got {confidence_threshold}"
            )

        if not isinstance(min_face_size, tuple) or len(min_face_size) != 2:
            raise InvalidParameterError(
                f"min_face_size must be a tuple of (width, height), got {min_face_size}"
            )

        if min_face_size[0] <= 0 or min_face_size[1] <= 0:
            raise InvalidParameterError(
                f"min_face_size dimensions must be positive, got {min_face_size}"
            )

        if nms_threshold < 0.0 or nms_threshold > 1.0:
            raise InvalidParameterError(
                f"nms_threshold must be between 0.0 and 1.0, got {nms_threshold}"
            )

        if max_cache_size < 0:
            raise InvalidParameterError(
                f"max_cache_size must be non-negative, got {max_cache_size}"
            )

    def _setup_logger(self, log_level: int) -> logging.Logger:
        """Setup a module logger.

        Library code should not install handlers by default; applications (and the
        demo script) should configure logging via `logging.basicConfig()` or a
        structured logging setup.
        """
        logger = logging.getLogger(__name__)
        logger.setLevel(log_level)
        return logger

    # Main API methods - delegate to components

    def detect_face(
        self, image_input: ImageInput
    ) -> List[Tuple[int, int, int, int, float]]:
        """Detect all faces in an image and return their coordinates with confidence scores."""
        return self.detector.detect(image_input)

    def detect_face_with_metrics(
        self, image_input: ImageInput
    ) -> DetectionMetrics:
        """Detect faces and return detailed metrics."""
        result = self.detector.detect_with_metrics(image_input)
        self.metrics.record_detection(
            result.processing_time, result.num_faces, result.cache_hit
        )
        return result

    def extract_faces(
        self,
        image_input: ImageInput,
        padding: float = 0.2,
    ) -> List[Tuple[NDArray[np.uint8], Tuple[int, int, int, int], float]]:
        """Detect and extract face images."""
        return self.detector.extract_faces(image_input, padding)

    def visualize_detections(
        self,
        image_input: ImageInput,
        faces: Optional[List[Tuple[int, int, int, int, float]]] = None,
        output_path: Optional[Path] = None,
        show_confidence: bool = True,
        color: Tuple[int, int, int] = (0, 255, 0),
        thickness: int = 2,
    ) -> NDArray[np.uint8]:
        """Draw bounding boxes on image."""
        img = self.image_loader.load(image_input)

        if faces is None:
            faces = self.detector.detect(image_input)

        result = self.visualizer.draw_detections(
            img, faces, show_confidence, color, thickness
        )

        if output_path:
            self.visualizer.save_visualization(result, output_path)

        return result

    def detect_faces_batch(
        self, image_paths: List[Union[Path, str]], show_progress: bool = True
    ) -> Dict[str, List[Tuple[int, int, int, int, float]]]:
        """Process multiple images in batch."""
        results: Dict[str, List[Tuple[int, int, int, int, float]]] = {}
        total = len(image_paths)

        for idx, path in enumerate(image_paths, 1):
            path = Path(path)
            try:
                faces = self.detect_face(path)
                results[str(path)] = faces
                if show_progress:
                    self.logger.info(
                        "Progress: %d/%d - %s: %d faces",
                        idx,
                        total,
                        path.name,
                        len(faces),
                    )
            except FaceDetectorError as e:  # pragma: no cover
                # Swallow *expected* library errors so the batch can continue.
                self.logger.error("Error processing %s: %s", path, e)
                results[str(path)] = []

        return results

    def clear_cache(self) -> None:
        """Clear the detection cache."""
        self.cache.clear()

    def get_cache_size(self) -> int:
        """Get number of cached results."""
        return self.cache.size()

    # Factory methods

    @classmethod
    def create_fast_detector(
        cls, model_dir: Optional[Path] = None, log_level: int = logging.WARNING
    ) -> "FaceDetectorDNN":
        """Factory method: Create a detector optimized for speed."""
        # Local import to avoid an import-time cycle.
        from .factory import FaceDetectorFactory

        return FaceDetectorFactory.create_fast_detector(model_dir, log_level)

    @classmethod
    def create_accurate_detector(
        cls, model_dir: Optional[Path] = None, log_level: int = logging.INFO
    ) -> "FaceDetectorDNN":
        """Factory method: Create a detector optimized for accuracy."""
        from .factory import FaceDetectorFactory

        return FaceDetectorFactory.create_accurate_detector(
            model_dir, log_level
        )

    @classmethod
    def create_balanced_detector(
        cls, model_dir: Optional[Path] = None, log_level: int = logging.INFO
    ) -> "FaceDetectorDNN":
        """Factory method: Create a detector with balanced speed/accuracy."""
        from .factory import FaceDetectorFactory

        return FaceDetectorFactory.create_balanced_detector(
            model_dir, log_level
        )
