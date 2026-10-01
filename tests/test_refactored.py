"""Smoke tests for the blitzid face detection package.

Validates:
- importability and basic construction
- deterministic default ``model_dir``
- bbox validity invariants (non-negative, within image bounds)
- cache hit behaviour and metrics timing semantics

Run::

    uv run pytest tests/ -v
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from blitzid import FaceDetectorDNN
from blitzid._models import default_model_dir

REPO_ROOT = Path(__file__).resolve().parent.parent

SYNTHETIC_IMAGE = np.zeros((480, 640, 3), dtype=np.uint8)


def _assert_bbox_valid(
    faces: list[tuple[int, int, int, int, float]],
    image_size: tuple[int, int],
) -> None:
    img_w, img_h = image_size
    for x, y, w, h, conf in faces:
        assert isinstance(x, int) and isinstance(y, int)
        assert isinstance(w, int) and isinstance(h, int)
        assert 0 <= x < img_w, (x, img_w)
        assert 0 <= y < img_h, (y, img_h)
        assert w > 0 and h > 0, (w, h)
        assert x + w <= img_w, (x, w, img_w)
        assert y + h <= img_h, (y, h, img_h)
        assert 0.0 <= float(conf) <= 1.0, conf


@pytest.fixture(scope="module")
def detector() -> FaceDetectorDNN:
    """Shared detector instance for the module (avoids re-loading the model)."""
    return FaceDetectorDNN(
        confidence_threshold=0.5,
        enable_cache=True,
        max_cache_size=16,
    )


def test_construction(detector: FaceDetectorDNN) -> None:
    """Detector can be constructed and reports a backend type."""
    assert detector.backend is not None


def test_deterministic_model_dir(detector: FaceDetectorDNN) -> None:
    """A detector without model_dir uses the default models directory."""
    actual = detector.model_manager.model_dir.resolve()
    assert actual == default_model_dir().resolve(), (
        f"Expected model_dir={default_model_dir()}, got {actual}"
    )


def test_synthetic_image_detection(detector: FaceDetectorDNN) -> None:
    """Detecting on a blank image returns valid (possibly empty) bboxes."""
    faces = detector.detect_face(SYNTHETIC_IMAGE)
    _assert_bbox_valid(faces, (SYNTHETIC_IMAGE.shape[1], SYNTHETIC_IMAGE.shape[0]))


def test_cache_and_metrics(detector: FaceDetectorDNN) -> None:
    """Second call on the same image is a cache hit with honest timing."""
    detector.clear_cache()

    r1 = detector.detect_face_with_metrics(SYNTHETIC_IMAGE)
    assert r1.cache_hit is False

    r2 = detector.detect_face_with_metrics(SYNTHETIC_IMAGE)
    assert r2.cache_hit is True
    assert r2.processing_time >= 0.0


def test_path_input(detector: FaceDetectorDNN) -> None:
    """Detection works when given a file path (requires repo image)."""
    image_path = REPO_ROOT / "images" / "bub_der_personalausweis_kopie.jpg"
    if not image_path.exists():
        pytest.skip(f"missing test image: {image_path}")

    import cv2

    raw = cv2.imread(str(image_path))
    assert raw is not None
    img_h, img_w = raw.shape[:2]

    faces = detector.detect_face(image_path)
    _assert_bbox_valid(faces, (img_w, img_h))
