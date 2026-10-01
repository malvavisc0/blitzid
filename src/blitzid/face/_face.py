"""Face record, metrics, and verification dataclasses for the face API."""

from __future__ import annotations

import hashlib
from collections import OrderedDict
from dataclasses import dataclass

import cv2
import numpy as np


@dataclass(frozen=True)
class Face:
    """A detected face.

    Attributes:
        bbox: ``(x, y, w, h)`` in image pixels.
        confidence: Detection score in ``[0, 1]``.
        landmarks: Subpixel ``(x, y)`` points in image pixels — right eye,
            left eye, nose tip, right mouth corner, left mouth corner
            (SCRFD order). Empty when no points are available.
    """

    bbox: tuple[int, int, int, int]
    confidence: float
    landmarks: tuple[tuple[float, float], ...] = ()


@dataclass
class DetectionMetrics:
    """Container for detection metrics.

    Attributes:
        faces: Detected faces as :class:`Face` records (sorted by
            confidence, descending).
        processing_time: Inference time in seconds (cache-lookup time on
            cache hits).
        image_size: ``(width, height)`` of the input image.
        backend: Backend identifier, e.g. ``"ONNXRuntime"``.
        num_faces: Number of detected faces.
        cache_hit: Whether the result came from the cache.
    """

    faces: list[Face]
    processing_time: float
    image_size: tuple[int, int]
    backend: str
    num_faces: int
    cache_hit: bool = False


@dataclass(frozen=True)
class VerificationResult:
    """Outcome of a 1:1 face verification.

    Attributes:
        verified: Whether the similarity met the decision threshold.
        similarity: Cosine similarity in ``[-1.0, 1.0]``.
        threshold: The decision threshold that was applied.
        processing_time: End-to-end verification time in seconds —
            image loading, face detection, embedding, and comparison.
        backend: Backend identifier, e.g. ``"ONNXRuntime"``.
    """

    verified: bool
    similarity: float
    threshold: float
    processing_time: float
    backend: str = "ONNXRuntime"


def as_tuple(face: Face) -> tuple[int, int, int, int, float]:
    """Project a :class:`Face` to the legacy (x, y, w, h, confidence) tuple."""
    x, y, w, h = face.bbox
    return (x, y, w, h, face.confidence)


def draw_detections(
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


def compute_hash(img: np.ndarray) -> str:
    """Content hash covering bytes, shape, and dtype.

    Shape and dtype are hashed alongside the pixel bytes so arrays that
    share byte content but differ in layout never collide on one key.
    """
    digest = hashlib.md5()
    digest.update(img.tobytes())
    digest.update(repr((img.shape, img.dtype)).encode())
    return digest.hexdigest()


class LRUCache:
    """Minimal LRU cache backed by :class:`~collections.OrderedDict`."""

    def __init__(self, max_size: int = 100):
        self.max_size = max_size
        self._data: OrderedDict[str, tuple[Face, ...]] = OrderedDict()

    def get(self, key: str) -> list[Face] | None:
        if key not in self._data:
            return None
        self._data.move_to_end(key)
        return list(self._data[key])

    def put(self, key: str, value: list[Face]) -> None:
        self._data[key] = tuple(value)
        self._data.move_to_end(key)
        while len(self._data) > self.max_size:
            self._data.popitem(last=False)

    def clear(self) -> None:
        self._data.clear()

    def size(self) -> int:
        return len(self._data)
