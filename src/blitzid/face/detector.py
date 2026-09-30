"""FaceDetectorDNN — the SCRFD-based face detector.

Runs the SCRFD-2.5G ONNX model via an onnxruntime CPU session, with an
optional LRU result cache, multi-scale detection, NMS + size filtering,
crop extraction, batch processing, and bbox visualization.
"""

from __future__ import annotations

import logging
import time
from pathlib import Path

import cv2
import numpy as np
from numpy.typing import NDArray

from .._image import ImageInput, load_image
from .._models import ModelManager, default_model_dir
from ..exceptions import BlitzIDError
from ._face import (
    DetectionMetrics,
    Face,
    LRUCache,
    as_tuple,
    compute_hash,
    draw_detections,
)
from ._scrfd import (
    SCRFD_STRIDES,
    anchor_centers,
    decode_outputs,
    letterbox_image,
    map_detections_to_faces,
    to_blob,
    validate_architecture,
)

__all__ = ["DetectionMetrics", "Face", "FaceDetectorDNN"]


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
    faces: list[Face],
    threshold: float,
) -> list[Face]:
    """Non-Maximum Suppression.  Assumes *faces* is sorted by confidence."""
    keep: list[Face] = []
    for face in faces:
        if all(_calculate_iou(face.bbox, k.bbox) <= threshold for k in keep):
            keep.append(face)
    return keep


def _filter_by_size(
    faces: list[Face],
    min_size: tuple[int, int],
) -> list[Face]:
    """Drop faces smaller than *min_size* (w, h)."""
    min_w, min_h = min_size
    return [f for f in faces if f.bbox[2] >= min_w and f.bbox[3] >= min_h]


def _normalize_scales(scales: tuple[float, ...]) -> tuple[float, ...]:
    """Validate, deduplicate, and sort multi-scale factors.

    The values are used exactly as given: passing ``scales=(1.5,)`` runs
    only the 1.5 pass, not a hidden 1.0 pass.
    """
    if not scales or any(s < 1.0 for s in scales):
        raise BlitzIDError("scales must be non-empty with values >= 1.0")
    return tuple(sorted(set(scales)))


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
        det_size: tuple[int, int] = (640, 640),
    ):
        self._validate_parameters(
            confidence_threshold, min_face_size, nms_threshold, max_cache_size, det_size
        )

        if model_dir is None:
            model_dir = default_model_dir()

        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(log_level)

        self.model_manager = ModelManager(model_dir, self.logger)
        self.session = self.model_manager.load_session()
        self.backend = "ONNXRuntime"

        # SCRFD ONNX export order: 3x scores (N,1), 3x boxes (N,4), 3x keypoints.
        self._input_name = self.session.get_inputs()[0].name
        self._output_names = [output.name for output in self.session.get_outputs()]
        validate_architecture(len(self._output_names))

        self.confidence_threshold = confidence_threshold
        self.min_face_size = min_face_size
        self.nms_threshold = nms_threshold
        self.multi_scale = bool(multi_scale)
        self.scales = _normalize_scales(scales)
        self.det_size = det_size

        det_w, det_h = det_size
        self._anchors = tuple(
            anchor_centers(det_h // stride, det_w // stride, stride)
            for stride in SCRFD_STRIDES
        )

        self._cache = LRUCache(max_cache_size) if enable_cache else None

    @classmethod
    def create_fast_detector(
        cls,
        model_dir: Path | None = None,
        log_level: int = logging.WARNING,
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
        )

    @classmethod
    def create_accurate_detector(
        cls,
        model_dir: Path | None = None,
        log_level: int = logging.INFO,
    ) -> FaceDetectorDNN:
        """Accuracy-optimized preset."""
        return cls(
            confidence_threshold=0.3,
            min_face_size=(20, 20),
            nms_threshold=0.2,
            log_level=log_level,
            enable_cache=False,
            model_dir=model_dir,
        )

    @classmethod
    def create_balanced_detector(
        cls,
        model_dir: Path | None = None,
        log_level: int = logging.INFO,
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
        )

    @staticmethod
    def _validate_det_size(det_size: tuple[int, int]) -> None:
        if not isinstance(det_size, tuple) or len(det_size) != 2:
            raise BlitzIDError(f"det_size must be (width, height), got {det_size}")
        if det_size[0] <= 0 or det_size[1] <= 0:
            raise BlitzIDError(f"det_size dimensions must be positive, got {det_size}")
        if det_size[0] % 32 != 0 or det_size[1] % 32 != 0:
            raise BlitzIDError(
                f"det_size dimensions must be multiples of 32, got {det_size}"
            )

    @staticmethod
    def _validate_parameters(
        confidence_threshold: float,
        min_face_size: tuple[int, int],
        nms_threshold: float,
        max_cache_size: int,
        det_size: tuple[int, int],
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
        FaceDetectorDNN._validate_det_size(det_size)

    def _detect_from_array(self, img: np.ndarray) -> tuple[list[Face], float, bool]:
        """Detect faces from an already-loaded array.

        Returns:
            ``(faces, processing_time, cache_hit)``. ``processing_time``
            is the real inference time; cache hits return the elapsed
            cache-lookup time so downstream averages stay honest.
        """
        start = time.time()
        if self._cache is not None:
            key = compute_hash(img)
            cached = self._cache.get(key)
            if cached is not None:
                self.logger.debug("Cache hit for key: %s...", key[:8])
                return cached, time.time() - start, True

        faces = self._postprocess(
            self._run_detection_multi_scale(img)
            if self.multi_scale
            else self._run_detection(img)
        )
        elapsed = time.time() - start

        if self._cache is not None:
            self._cache.put(key, faces)
        return faces, elapsed, False

    def detect_face(
        self, image_input: ImageInput
    ) -> list[tuple[int, int, int, int, float]]:
        """Detect faces.  Returns list of (x, y, w, h, confidence)."""
        return [as_tuple(face) for face in self.detect_face_landmarks(image_input)]

    def detect_face_landmarks(self, image_input: ImageInput) -> list[Face]:
        """Detect faces with landmarks.  Returns list of :class:`Face`."""
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
        for face in faces:
            x, y, fw, fh = face.bbox
            pad_w = int(fw * padding)
            pad_h = int(fh * padding)
            x1 = max(0, x - pad_w)
            y1 = max(0, y - pad_h)
            x2 = min(w, x + fw + pad_w)
            y2 = min(h, y + fh + pad_h)
            extracted.append((img[y1:y2, x1:x2], (x, y, fw, fh), face.confidence))

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
            detected, _, _ = self._detect_from_array(img)
            faces = [as_tuple(face) for face in detected]

        result = draw_detections(img, faces, show_confidence, color, thickness)

        if output_path:
            cv2.imwrite(str(output_path), result)
            self.logger.info("Visualization saved to %s", output_path)

        return result

    def detect_faces_batch(
        self,
        image_paths: list[Path | str],
        show_progress: bool = True,
    ) -> dict[str, list[tuple[int, int, int, int, float]]]:
        """Process multiple images in batch.

        Failed images are omitted from the result dict (an empty list
        always means "processed, zero faces").
        """
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

        return results

    def clear_cache(self) -> None:
        """Clear the detection cache."""
        if self._cache is not None:
            self._cache.clear()

    def get_cache_size(self) -> int:
        """Get number of cached results."""
        return self._cache.size() if self._cache is not None else 0

    @property
    def cache_enabled(self) -> bool:
        """Whether result caching is enabled on this detector."""
        return self._cache is not None

    def _run_detection(self, img: np.ndarray) -> list[Face]:
        """SCRFD forward pass on a single-scale image."""
        h, w = img.shape[:2]
        letterboxed, scale = letterbox_image(img, self.det_size)
        blob = to_blob(letterboxed)
        net_outs = self.session.run(self._output_names, {self._input_name: blob})

        detections = decode_outputs(
            net_outs[:3],
            net_outs[3:6],
            net_outs[6:9],
            self._anchors,
            self.confidence_threshold,
        )
        return map_detections_to_faces(detections, scale, (w, h))

    def _run_detection_multi_scale(self, img: np.ndarray) -> list[Face]:
        """Run detection across multiple scales and map boxes back."""
        orig_h, orig_w = img.shape[:2]
        all_faces: list[Face] = []

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

            for face in faces_scaled:
                x, y, bw, bh = face.bbox
                x0 = int(max(0, min(round(x / s), orig_w - 1)))
                y0 = int(max(0, min(round(y / s), orig_h - 1)))
                w0 = int(max(1, min(round(bw / s), orig_w - x0)))
                h0 = int(max(1, min(round(bh / s), orig_h - y0)))
                points = tuple(
                    (
                        max(0, min(round(px / s), orig_w - 1)),
                        max(0, min(round(py / s), orig_h - 1)),
                    )
                    for px, py in face.landmarks
                )
                all_faces.append(
                    Face(
                        bbox=(x0, y0, w0, h0),
                        confidence=face.confidence,
                        landmarks=points,
                    )
                )

        return all_faces

    def _postprocess(self, faces: list[Face]) -> list[Face]:
        """Size filter → sort → NMS."""
        faces = _filter_by_size(faces, self.min_face_size)
        faces = sorted(faces, key=lambda f: f.confidence, reverse=True)
        faces = _apply_nms(faces, self.nms_threshold)
        return faces

    def _log_results(
        self,
        image_input: ImageInput,
        faces: list[Face],
    ) -> None:
        if isinstance(image_input, Path):
            image_name = image_input.name
        elif isinstance(image_input, str):
            image_name = Path(image_input).name
        else:
            image_name = "image"

        if faces:
            self.logger.info("%d face(s) detected in %s", len(faces), image_name)
            for idx, face in enumerate(faces, 1):
                x, y, wb, hb = face.bbox
                self.logger.debug(
                    "Face %d: confidence=%.2f | (%d,%d,%dx%d)",
                    idx,
                    face.confidence,
                    x,
                    y,
                    wb,
                    hb,
                )
        else:
            self.logger.info("No faces detected: %s", image_name)
