"""Core face detection logic."""

import logging
import time
from pathlib import Path
from typing import List, Optional, Tuple, Union

import cv2
import numpy as np

try:
    from PIL import Image

    PIL_AVAILABLE = True
except ImportError:
    PIL_AVAILABLE = False

from .cache import DetectionCache
from .exceptions import InvalidParameterError
from .image_loader import ImageLoader
from .metrics import DetectionMetrics
from .models import ModelManager
from .processors import FaceProcessor


class FaceDetector:
    """Core face detection using DNN."""

    def __init__(
        self,
        model_manager: ModelManager,
        image_loader: ImageLoader,
        processor: FaceProcessor,
        cache: DetectionCache,
        confidence_threshold: float = 0.5,
        min_face_size: Tuple[int, int] = (50, 50),
        logger: Optional[logging.Logger] = None,
        multi_scale: bool = False,
        scales: Tuple[float, ...] = (1.0, 1.5, 2.0),
        use_cuda: bool = True,
        require_cuda: bool = False,
    ):
        """
        Initialize face detector.

        Args:
            model_manager: Model manager instance
            image_loader: Image loader instance
            processor: Face processor instance
            cache: Detection cache instance
            confidence_threshold: Minimum confidence for face detection (0.0-1.0)
            min_face_size: Minimum face size as (width, height) in pixels
            logger: Optional logger instance
        """
        self.model_manager = model_manager
        self.image_loader = image_loader
        self.processor = processor
        self.cache = cache
        self.confidence_threshold = confidence_threshold
        self.min_face_size = min_face_size
        self.logger = logger or logging.getLogger(__name__)

        self.multi_scale = bool(multi_scale)
        self.scales = self._normalize_scales(scales)

        # Load network
        self.net, self.backend = self.model_manager.load_network(
            use_cuda=use_cuda, require_cuda=require_cuda
        )

    @staticmethod
    def _normalize_scales(scales: Tuple[float, ...]) -> Tuple[float, ...]:
        """Normalize and validate multi-scale factors.

        Rules:
        - must be a non-empty tuple of finite floats
        - must contain 1.0 (baseline)
        - values must be >= 1.0 (only upscale for recall improvement)

        Args:
            scales: Tuple of scale factors for multi-scale detection

        Returns:
            Normalized, deduplicated, and sorted tuple of scale factors

        Raises:
            InvalidParameterError: If scales is invalid or contains invalid values

        Examples:
            >>> FaceDetector._normalize_scales((1.5, 2.0))
            (1.0, 1.5, 2.0)  # 1.0 auto-added as baseline

            >>> FaceDetector._normalize_scales((1.0, 1.0, 1.5))
            (1.0, 1.5)  # duplicates removed

            >>> FaceDetector._normalize_scales((2.0, 1.5, 1.0))
            (1.0, 1.5, 2.0)  # sorted for deterministic cache keys

            >>> FaceDetector._normalize_scales((0.5, 1.0))
            InvalidParameterError: Scale must be >= 1.0 (upscale only), got 0.5

            >>> FaceDetector._normalize_scales(())
            InvalidParameterError: scales must be a non-empty tuple of floats
        """
        if not isinstance(scales, tuple) or len(scales) == 0:
            raise InvalidParameterError(
                f"scales must be a non-empty tuple of floats, got {scales}"
            )

        normalized: list[float] = []
        for s in scales:
            try:
                sf = float(s)
            except (TypeError, ValueError) as e:
                raise InvalidParameterError(
                    f"Invalid scale value {s!r}: {e}"
                ) from e

            if not np.isfinite(sf):
                raise InvalidParameterError(f"Scale must be finite, got {sf}")
            if sf < 1.0:
                raise InvalidParameterError(
                    f"Scale must be >= 1.0 (upscale only), got {sf}"
                )
            normalized.append(sf)

        # Always include 1.0 for stable behavior.
        if all(abs(sf - 1.0) > 1e-9 for sf in normalized):
            normalized.append(1.0)

        # Deduplicate + sort for deterministic cache keys.
        unique = sorted({round(sf, 6) for sf in normalized})
        return tuple(float(sf) for sf in unique)

    def _build_cache_key(self, img: np.ndarray) -> str:
        """Build a cache key that includes detector configuration."""
        base = self.cache.compute_hash(img)
        scales = ",".join(f"{s:.2f}" for s in self.scales)
        return (
            f"{base}|ct={self.confidence_threshold:.3f}"
            f"|mfs={self.min_face_size[0]}x{self.min_face_size[1]}"
            f"|nms={float(self.processor.nms_threshold):.3f}"
            f"|ms={1 if self.multi_scale else 0}|scales={scales}"
        )

    def _run_detection_multi_scale(
        self, img: np.ndarray
    ) -> List[Tuple[int, int, int, int, float]]:
        """Run detection on multiple upscaled versions and map boxes back."""
        orig_h, orig_w = img.shape[:2]
        all_faces: list[Tuple[int, int, int, int, float]] = []

        for scale in self.scales:
            if abs(scale - 1.0) < 1e-9:
                scaled = img
                s = 1.0
            else:
                new_w = max(1, int(round(orig_w * scale)))
                new_h = max(1, int(round(orig_h * scale)))
                scaled = cv2.resize(
                    img, (new_w, new_h), interpolation=cv2.INTER_LINEAR
                )
                s = float(scale)

            faces_scaled = self._run_detection(scaled)
            if abs(s - 1.0) < 1e-9:
                all_faces.extend(faces_scaled)
                continue

            # Map back to original coordinates.
            for x, y, w_box, h_box, conf in faces_scaled:
                x0 = int(round(x / s))
                y0 = int(round(y / s))
                w0 = int(round(w_box / s))
                h0 = int(round(h_box / s))

                # Clamp and drop degenerate.
                x0 = int(max(0, min(x0, orig_w - 1)))
                y0 = int(max(0, min(y0, orig_h - 1)))
                w0 = int(max(1, min(w0, orig_w - x0)))
                h0 = int(max(1, min(h0, orig_h - y0)))

                all_faces.append((x0, y0, w0, h0, conf))

        return all_faces

    def detect(
        self, image_input: Union[Path, str, np.ndarray, "Image.Image"]
    ) -> List[Tuple[int, int, int, int, float]]:
        """
        Detect faces in image.

        Args:
            image_input: Path to image file, numpy array, or PIL Image

        Returns:
            List of tuples (x, y, w, h, confidence) for each detected face,
            sorted by confidence in descending order
        """
        # Load image
        img = self.image_loader.load(image_input)

        cache_key: Optional[str] = None

        # Check cache
        if self.cache.enabled:
            cache_key = self._build_cache_key(img)
            cached = self.cache.get(cache_key)
            if cached is not None:
                return cached

        # Run detection
        if self.multi_scale:
            faces = self._run_detection_multi_scale(img)
        else:
            faces = self._run_detection(img)

        # Post-process (merges duplicates across scales via NMS)
        faces = self._postprocess_faces(faces)

        # Cache result
        if self.cache.enabled and cache_key is not None:
            self.cache.put(cache_key, faces)

        # Log results
        self._log_results(image_input, faces)

        return faces

    def detect_with_metrics(
        self, image_input: Union[Path, str, np.ndarray, "Image.Image"]
    ) -> DetectionMetrics:
        """Detect faces and return detailed metrics."""
        img = self.image_loader.load(image_input)
        h, w = img.shape[:2]

        cache_hit = False
        cache_key: Optional[str] = None

        # Cache check (excluded from `processing_time` on cache miss; cache hit is 0.0)
        if self.cache.enabled:
            cache_key = self._build_cache_key(img)
            cached = self.cache.get(cache_key)
            if cached is not None:
                cache_hit = True
                faces = cached
                processing_time = 0.0
            else:
                start_time = time.time()
                if self.multi_scale:
                    faces = self._postprocess_faces(
                        self._run_detection_multi_scale(img)
                    )
                else:
                    faces = self._postprocess_faces(self._run_detection(img))
                processing_time = time.time() - start_time

                self.cache.put(cache_key, faces)
        else:
            start_time = time.time()
            if self.multi_scale:
                faces = self._postprocess_faces(
                    self._run_detection_multi_scale(img)
                )
            else:
                faces = self._postprocess_faces(self._run_detection(img))
            processing_time = time.time() - start_time

        # Keep behavior consistent with `detect()` (logs once per call).
        self._log_results(image_input, faces)

        return DetectionMetrics(
            faces=faces,
            processing_time=processing_time,
            image_size=(w, h),
            backend=self.backend,
            num_faces=len(faces),
            cache_hit=cache_hit,
        )

    def extract_faces(
        self,
        image_input: Union[Path, str, np.ndarray, "Image.Image"],
        padding: float = 0.2,
    ) -> List[Tuple[np.ndarray, Tuple[int, int, int, int], float]]:
        """
        Detect and extract face images.

        Args:
            image_input: Image to process
            padding: Percentage padding around face (0.2 = 20%)

        Returns:
            List of (face_image, bbox, confidence) tuples
        """
        if padding < 0.0 or padding > 1.0:
            raise InvalidParameterError(
                f"padding must be between 0.0 and 1.0, got {padding}"
            )

        img = self.image_loader.load(image_input)
        faces = self.detect(image_input)

        h, w = img.shape[:2]
        extracted = []

        for x, y, face_w, face_h, conf in faces:
            # Calculate padding
            pad_w = int(face_w * padding)
            pad_h = int(face_h * padding)

            # Apply padding with bounds checking
            x1 = max(0, x - pad_w)
            y1 = max(0, y - pad_h)
            x2 = min(w, x + face_w + pad_w)
            y2 = min(h, y + face_h + pad_h)

            # Extract face
            face_img = img[y1:y2, x1:x2]
            extracted.append((face_img, (x, y, face_w, face_h), conf))

        return extracted

    def _postprocess_faces(
        self, faces: List[Tuple[int, int, int, int, float]]
    ) -> List[Tuple[int, int, int, int, float]]:
        """Apply size filtering, sorting, and NMS."""
        faces = self.processor.filter_by_size(faces, self.min_face_size)
        faces = self.processor.sort_by_confidence(faces)
        faces = self.processor.apply_nms(faces)
        return faces

    def _run_detection(
        self, img: np.ndarray
    ) -> List[Tuple[int, int, int, int, float]]:
        """
        Run DNN forward pass.

        Args:
            img: Image as numpy array

        Returns:
            List of detected faces (x, y, w, h, confidence)
        """
        h, w = img.shape[:2]
        blob = cv2.dnn.blobFromImage(
            image=img,
            scalefactor=1.0,
            size=(300, 300),
            mean=(104.0, 177.0, 123.0),
            swapRB=False,
            crop=False,
        )

        self.net.setInput(blob)
        detections = self.net.forward()

        faces = []
        for i in range(detections.shape[2]):
            conf = float(detections[0, 0, i, 2])
            if conf <= self.confidence_threshold:
                continue

            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            x1, y1, x2, y2 = box.astype(int)

            # Clamp to image bounds and drop invalid/degenerate boxes.
            # Convert to plain Python ints for a stable public API.
            x1 = int(max(0, min(int(x1), w - 1)))
            y1 = int(max(0, min(int(y1), h - 1)))
            x2 = int(max(0, min(int(x2), w)))
            y2 = int(max(0, min(int(y2), h)))

            if x2 <= x1 or y2 <= y1:
                continue

            width, height = int(x2 - x1), int(y2 - y1)
            faces.append((x1, y1, width, height, conf))

        return faces

    def _log_results(
        self,
        image_input: Union[Path, str, np.ndarray, "Image.Image"],
        faces: List[Tuple[int, int, int, int, float]],
    ) -> None:
        """
        Log detection results.

        Args:
            image_input: Original image input
            faces: Detected faces
        """
        if faces:
            image_name = (
                image_input.name if isinstance(image_input, Path) else "image"
            )
            self.logger.info(
                "%d face(s) detected in %s", len(faces), image_name
            )
            for idx, (x, y, w_box, h_box, conf) in enumerate(faces, 1):
                self.logger.debug(
                    "Face %d: confidence=%.2f | (%d,%d,%dx%d)",
                    idx,
                    conf,
                    x,
                    y,
                    w_box,
                    h_box,
                )
        else:
            image_name = (
                image_input.name if isinstance(image_input, Path) else "image"
            )
            self.logger.info("No faces detected: %s", image_name)
