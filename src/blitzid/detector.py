"""FaceDetectorDNN — the single public DNN detection class.

Merges the former facade, core detector, factory presets, cache, post-processing,
visualizer, and metrics dataclass into one cohesive module.

Design changes vs. the original multi-file architecture:
- One class, no two-layer delegation.
- NMS / IoU / size-filter are plain private functions.
- Cache is an optional ``OrderedDict`` with ``get / put / clear``.
- ``DetectionMetrics`` dataclass is the only return type from metrics methods.
- Factory presets live as ``@classmethod`` helpers on this class.
- CLAHE preprocessing removed; images are loaded raw.
"""

from __future__ import annotations

import hashlib
import logging
import time
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
from numpy.typing import NDArray

from ._image import ImageInput, load_image
from ._models import ModelManager, default_model_dir
from .exceptions import BlitzIDError

# ---------------------------------------------------------------------------
# Public dataclass (return type for metrics methods)
# ---------------------------------------------------------------------------


@dataclass
class DetectionMetrics:
    """Container for detection metrics."""

    faces: list[tuple[int, int, int, int, float]]
    processing_time: float
    image_size: tuple[int, int]
    backend: str
    num_faces: int
    cache_hit: bool = False


# ---------------------------------------------------------------------------
# Private helpers — NMS, IoU, size filtering (replaces FaceProcessor)
# ---------------------------------------------------------------------------


def _calculate_iou(
    box1: tuple[int, int, int, int],
    box2: tuple[int, int, int, int],
) -> float:
    """Intersection over Union between two (x, y, w, h) boxes."""
    x1, y1, w1, h1 = box1
    x2, y2, w2, h2 = box2

    x_left = max(x1, x2)
    y_top = max(y1, y2)
    x_right = min(x1 + w1, x2 + w2)
    y_bottom = min(y1 + h1, y2 + h2)

    if x_right < x_left or y_bottom < y_top:
        return 0.0

    intersection = (x_right - x_left) * (y_bottom - y_top)
    union = w1 * h1 + w2 * h2 - intersection
    return intersection / union if union > 0 else 0.0


def _apply_nms(
    faces: list[tuple[int, int, int, int, float]],
    threshold: float,
) -> list[tuple[int, int, int, int, float]]:
    """Non-Maximum Suppression.  Assumes *faces* is sorted by confidence."""
    keep: list[tuple[int, int, int, int, float]] = []
    for face in faces:
        if all(_calculate_iou(face[:4], k[:4]) <= threshold for k in keep):
            keep.append(face)
    return keep


def _filter_by_size(
    faces: list[tuple[int, int, int, int, float]],
    min_size: tuple[int, int],
) -> list[tuple[int, int, int, int, float]]:
    """Drop faces smaller than *min_size* (w, h)."""
    min_w, min_h = min_size
    return [f for f in faces if f[2] >= min_w and f[3] >= min_h]


# ---------------------------------------------------------------------------
# Private helpers — visualizer (replaces FaceVisualizer)
# ---------------------------------------------------------------------------


def _draw_detections(
    image: np.ndarray,
    faces: list[tuple[int, int, int, int, float]],
    show_confidence: bool = True,
    color: tuple[int, int, int] = (0, 255, 0),
    thickness: int = 2,
) -> np.ndarray:
    """Draw bounding boxes (and optional labels) on a *copy* of *image*."""
    img = image.copy()
    for x, y, w, h, conf in faces:
        cv2.rectangle(img, (x, y), (x + w, y + h), color, thickness)
        if show_confidence:
            label = f"{conf:.2f}"
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            ly = max(y, th + 10)
            cv2.rectangle(img, (x, ly - th - 10), (x + tw, ly), color, -1)
            cv2.putText(
                img,
                label,
                (x, ly - 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 255, 255),
                1,
            )
    return img


# ---------------------------------------------------------------------------
# Private helpers — cache (replaces DetectionCache)
# ---------------------------------------------------------------------------


def _compute_hash(img: np.ndarray) -> str:
    """Content hash for an image array.  Raises on failure instead of
    silently falling back to shape/dtype (Bug 5 fix)."""
    return hashlib.md5(img.tobytes()).hexdigest()


class _LRUCache:
    """Minimal LRU cache backed by :class:`~collections.OrderedDict`."""

    def __init__(self, max_size: int = 100):
        self.max_size = max_size
        self._data: OrderedDict[str, tuple[tuple[int, int, int, int, float], ...]] = (
            OrderedDict()
        )

    def get(self, key: str) -> list[tuple[int, int, int, int, float]] | None:
        if key not in self._data:
            return None
        self._data.move_to_end(key)
        return list(self._data[key])

    def put(self, key: str, value: list[tuple[int, int, int, int, float]]) -> None:
        self._data[key] = tuple(value)
        self._data.move_to_end(key)
        while len(self._data) > self.max_size:
            self._data.popitem(last=False)

    def clear(self) -> None:
        self._data.clear()

    def size(self) -> int:
        return len(self._data)


# ---------------------------------------------------------------------------
# Scale normalisation (simplified)
# ---------------------------------------------------------------------------


def _normalize_scales(scales: tuple[float, ...]) -> tuple[float, ...]:
    """Validate and normalize multi-scale factors."""
    if not scales or any(s < 1.0 for s in scales):
        raise BlitzIDError("scales must be non-empty with values >= 1.0")
    return tuple(sorted(set(scales) | {1.0}))


# ---------------------------------------------------------------------------
# FaceDetectorDNN  — the public API
# ---------------------------------------------------------------------------


class FaceDetectorDNN:
    """DNN-based face detector.

    Provides detection, extraction, visualization, and batch processing
    with an optional LRU cache.
    """

    def __init__(
        self,
        confidence_threshold: float = 0.5,
        min_face_size: tuple[int, int] = (50, 50),
        nms_threshold: float = 0.3,
        log_level: int = logging.INFO,
        enable_cache: bool = False,
        max_cache_size: int = 100,
        model_dir: Path | None = None,
        multi_scale: bool = False,
        scales: tuple[float, ...] = (1.0, 1.5, 2.0),
        require_cuda: bool = False,
    ):
        # --- validate ---
        self._validate_parameters(
            confidence_threshold, min_face_size, nms_threshold, max_cache_size
        )

        if model_dir is None:
            model_dir = default_model_dir()

        # --- logger ---
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(log_level)

        # --- model ---
        self.model_manager = ModelManager(model_dir, self.logger)
        self.net, self.backend = self.model_manager.load_network(
            use_cuda=True, require_cuda=require_cuda
        )

        # --- config ---
        self.confidence_threshold = confidence_threshold
        self.min_face_size = min_face_size
        self.nms_threshold = nms_threshold
        self.multi_scale = bool(multi_scale)
        self.scales = _normalize_scales(scales)
        self.require_cuda = bool(require_cuda)

        # --- optional cache ---
        self._cache = _LRUCache(max_cache_size) if enable_cache else None

        # Backward-compat alias (old facade exposed ``backend_type``).
        self.backend_type = self.backend

    # ------------------------------------------------------------------
    # Factory presets (replaces FaceDetectorFactory)
    # ------------------------------------------------------------------

    @classmethod
    def create_fast_detector(
        cls,
        model_dir: Path | None = None,
        log_level: int = logging.WARNING,
        require_cuda: bool = False,
    ) -> FaceDetectorDNN:
        """Speed-optimized preset."""
        return cls(
            confidence_threshold=0.7,
            min_face_size=(80, 80),
            nms_threshold=0.4,
            log_level=log_level,
            enable_cache=True,
            max_cache_size=200,
            model_dir=model_dir,
            require_cuda=require_cuda,
        )

    @classmethod
    def create_accurate_detector(
        cls,
        model_dir: Path | None = None,
        log_level: int = logging.INFO,
        require_cuda: bool = False,
    ) -> FaceDetectorDNN:
        """Accuracy-optimized preset."""
        return cls(
            confidence_threshold=0.3,
            min_face_size=(20, 20),
            nms_threshold=0.2,
            log_level=log_level,
            enable_cache=False,
            model_dir=model_dir,
            require_cuda=require_cuda,
        )

    @classmethod
    def create_balanced_detector(
        cls,
        model_dir: Path | None = None,
        log_level: int = logging.INFO,
        require_cuda: bool = False,
    ) -> FaceDetectorDNN:
        """Balanced speed/accuracy preset."""
        return cls(
            confidence_threshold=0.5,
            min_face_size=(50, 50),
            nms_threshold=0.3,
            log_level=log_level,
            enable_cache=True,
            max_cache_size=100,
            model_dir=model_dir,
            require_cuda=require_cuda,
        )

    # ------------------------------------------------------------------
    # Parameter validation
    # ------------------------------------------------------------------

    @staticmethod
    def _validate_parameters(
        confidence_threshold: float,
        min_face_size: tuple[int, int],
        nms_threshold: float,
        max_cache_size: int,
    ) -> None:
        if not (0.0 <= confidence_threshold <= 1.0):
            raise BlitzIDError(
                f"confidence_threshold must be between 0.0 and 1.0, "
                f"got {confidence_threshold}"
            )
        if not isinstance(min_face_size, tuple) or len(min_face_size) != 2:
            raise BlitzIDError(
                f"min_face_size must be (width, height), got {min_face_size}"
            )
        if min_face_size[0] <= 0 or min_face_size[1] <= 0:
            raise BlitzIDError(
                f"min_face_size dimensions must be positive, got {min_face_size}"
            )
        if not (0.0 <= nms_threshold <= 1.0):
            raise BlitzIDError(
                f"nms_threshold must be between 0.0 and 1.0, got {nms_threshold}"
            )
        if max_cache_size < 0:
            raise BlitzIDError(
                f"max_cache_size must be non-negative, got {max_cache_size}"
            )

    # ------------------------------------------------------------------
    # Core detection from array (single path for cache + detect + NMS)
    # ------------------------------------------------------------------

    def _detect_from_array(
        self, img: np.ndarray
    ) -> tuple[list[tuple[int, int, int, int, float]], float, bool]:
        """Detect faces from an already-loaded array.

        Returns:
            ``(faces, processing_time, cache_hit)``
        """
        if self._cache is not None:
            key = _compute_hash(img)
            cached = self._cache.get(key)
            if cached is not None:
                self.logger.debug("Cache hit for key: %s...", key[:8])
                return cached, 0.0, True

            start = time.time()
            faces = self._postprocess(
                self._run_detection_multi_scale(img)
                if self.multi_scale
                else self._run_detection(img)
            )
            elapsed = time.time() - start
            self._cache.put(key, faces)
            return faces, elapsed, False

        start = time.time()
        faces = self._postprocess(
            self._run_detection_multi_scale(img)
            if self.multi_scale
            else self._run_detection(img)
        )
        elapsed = time.time() - start
        return faces, elapsed, False

    # ------------------------------------------------------------------
    # Public detection API
    # ------------------------------------------------------------------

    def detect_face(
        self, image_input: ImageInput
    ) -> list[tuple[int, int, int, int, float]]:
        """Detect faces.  Returns list of (x, y, w, h, confidence)."""
        img = load_image(image_input, self.logger)
        faces, _, _ = self._detect_from_array(img)
        self._log_results(image_input, faces)
        return faces

    def detect_face_with_metrics(self, image_input: ImageInput) -> DetectionMetrics:
        """Detect faces and return detailed metrics."""
        img = load_image(image_input, self.logger)
        h, w = img.shape[:2]

        faces, processing_time, cache_hit = self._detect_from_array(img)
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
        image_input: ImageInput,
        padding: float = 0.2,
    ) -> list[tuple[NDArray[np.uint8], tuple[int, int, int, int], float]]:
        """Detect and extract face images (single load, no double I/O)."""
        if not (0.0 <= padding <= 1.0):
            raise BlitzIDError(f"padding must be between 0.0 and 1.0, got {padding}")

        img = load_image(image_input, self.logger)
        faces, _, _ = self._detect_from_array(img)

        h, w = img.shape[:2]
        extracted: list[tuple[NDArray[np.uint8], tuple[int, int, int, int], float]] = []
        for x, y, fw, fh, conf in faces:
            pad_w = int(fw * padding)
            pad_h = int(fh * padding)
            x1 = max(0, x - pad_w)
            y1 = max(0, y - pad_h)
            x2 = min(w, x + fw + pad_w)
            y2 = min(h, y + fh + pad_h)
            extracted.append((img[y1:y2, x1:x2], (x, y, fw, fh), conf))

        return extracted

    def visualize_detections(
        self,
        image_input: ImageInput,
        faces: list[tuple[int, int, int, int, float]] | None = None,
        output_path: Path | None = None,
        show_confidence: bool = True,
        color: tuple[int, int, int] = (0, 255, 0),
        thickness: int = 2,
    ) -> NDArray[np.uint8]:
        """Draw bounding boxes on image (single load, no double I/O)."""
        img = load_image(image_input, self.logger)

        if faces is None:
            faces, _, _ = self._detect_from_array(img)

        result = _draw_detections(img, faces, show_confidence, color, thickness)

        if output_path:
            cv2.imwrite(str(output_path), result)
            self.logger.info("Visualization saved to %s", output_path)

        return result

    def detect_faces_batch(
        self,
        image_paths: list[Path | str],
        show_progress: bool = True,
    ) -> dict[str, list[tuple[int, int, int, int, float]]]:
        """Process multiple images in batch."""
        results: dict[str, list[tuple[int, int, int, int, float]]] = {}
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
            except BlitzIDError as e:  # pragma: no cover
                self.logger.error("Error processing %s: %s", path, e)
                results[str(path)] = []

        return results

    def clear_cache(self) -> None:
        """Clear the detection cache."""
        if self._cache is not None:
            self._cache.clear()

    def get_cache_size(self) -> int:
        """Get number of cached results."""
        return self._cache.size() if self._cache is not None else 0

    # ------------------------------------------------------------------
    # Internal detection
    # ------------------------------------------------------------------

    def _run_detection(self, img: np.ndarray) -> list[tuple[int, int, int, int, float]]:
        """DNN forward pass on a single-scale image."""
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

        faces: list[tuple[int, int, int, int, float]] = []
        for i in range(detections.shape[2]):
            conf = float(detections[0, 0, i, 2])
            if conf <= self.confidence_threshold:
                continue

            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            x1, y1, x2, y2 = box.astype(int)

            x1 = int(max(0, min(int(x1), w - 1)))
            y1 = int(max(0, min(int(y1), h - 1)))
            x2 = int(max(0, min(int(x2), w)))
            y2 = int(max(0, min(int(y2), h)))

            if x2 <= x1 or y2 <= y1:
                continue

            faces.append((x1, y1, int(x2 - x1), int(y2 - y1), conf))

        return faces

    def _run_detection_multi_scale(
        self, img: np.ndarray
    ) -> list[tuple[int, int, int, int, float]]:
        """Run detection across multiple scales and map boxes back."""
        orig_h, orig_w = img.shape[:2]
        all_faces: list[tuple[int, int, int, int, float]] = []

        for scale in self.scales:
            if abs(scale - 1.0) < 1e-9:
                scaled = img
                s = 1.0
            else:
                new_w = max(1, round(orig_w * scale))
                new_h = max(1, round(orig_h * scale))
                scaled = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
                s = float(scale)

            faces_scaled = self._run_detection(scaled)

            if abs(s - 1.0) < 1e-9:
                all_faces.extend(faces_scaled)
                continue

            for x, y, bw, bh, conf in faces_scaled:
                x0 = int(max(0, min(round(x / s), orig_w - 1)))
                y0 = int(max(0, min(round(y / s), orig_h - 1)))
                w0 = int(max(1, min(round(bw / s), orig_w - x0)))
                h0 = int(max(1, min(round(bh / s), orig_h - y0)))
                all_faces.append((x0, y0, w0, h0, conf))

        return all_faces

    def _postprocess(
        self, faces: list[tuple[int, int, int, int, float]]
    ) -> list[tuple[int, int, int, int, float]]:
        """Size filter → sort → NMS."""
        faces = _filter_by_size(faces, self.min_face_size)
        faces = sorted(faces, key=lambda f: f[4], reverse=True)
        faces = _apply_nms(faces, self.nms_threshold)
        return faces

    # ------------------------------------------------------------------
    # Logging
    # ------------------------------------------------------------------

    def _log_results(
        self,
        image_input: ImageInput,
        faces: list[tuple[int, int, int, int, float]],
    ) -> None:
        if isinstance(image_input, Path):
            image_name = image_input.name
        elif isinstance(image_input, str):
            image_name = Path(image_input).name
        else:
            image_name = "image"

        if faces:
            self.logger.info("%d face(s) detected in %s", len(faces), image_name)
            for idx, (x, y, wb, hb, conf) in enumerate(faces, 1):
                self.logger.debug(
                    "Face %d: confidence=%.2f | (%d,%d,%dx%d)",
                    idx,
                    conf,
                    x,
                    y,
                    wb,
                    hb,
                )
        else:
            self.logger.info("No faces detected: %s", image_name)
