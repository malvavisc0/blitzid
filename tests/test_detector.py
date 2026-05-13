"""Tests for FaceDetectorDNN — parameter validation + behavioural tests.

Parameter validation tests are fast (no model load).
Behavioural tests require the model to be available.
"""

from __future__ import annotations

import logging
import tempfile
from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import pytest

from blitzid.detector import DetectionMetrics, FaceDetectorDNN
from blitzid.exceptions import BlitzIDError

SYNTHETIC_IMAGE = np.zeros((480, 640, 3), dtype=np.uint8)


# ── Parameter validation (no model needed) ────────────────────


class TestParameterValidation:
    def test_invalid_confidence_threshold(self) -> None:
        with pytest.raises(BlitzIDError, match="confidence_threshold"):
            FaceDetectorDNN._validate_parameters(1.5, (50, 50), 0.3, 100)

    def test_invalid_nms_threshold(self) -> None:
        with pytest.raises(BlitzIDError, match="nms_threshold"):
            FaceDetectorDNN._validate_parameters(0.5, (50, 50), -0.1, 100)

    def test_invalid_min_face_size_negative(self) -> None:
        with pytest.raises(BlitzIDError, match="positive"):
            FaceDetectorDNN._validate_parameters(0.5, (-1, 50), 0.3, 100)

    def test_invalid_min_face_size_type(self) -> None:
        with pytest.raises(BlitzIDError, match="min_face_size"):
            FaceDetectorDNN._validate_parameters(0.5, [50, 50], 0.3, 100)  # type: ignore[arg-type]

    def test_invalid_max_cache_size(self) -> None:
        with pytest.raises(BlitzIDError, match="max_cache_size"):
            FaceDetectorDNN._validate_parameters(0.5, (50, 50), 0.3, -1)

    def test_boundary_confidence_0_valid(self) -> None:
        FaceDetectorDNN._validate_parameters(0.0, (50, 50), 0.3, 100)

    def test_boundary_confidence_1_valid(self) -> None:
        FaceDetectorDNN._validate_parameters(1.0, (50, 50), 0.3, 100)

    def test_boundary_cache_size_0_valid(self) -> None:
        FaceDetectorDNN._validate_parameters(0.5, (50, 50), 0.3, 0)


# ── extract_faces padding validation ──────────────────────────


class TestExtractFacesPaddingValidation:
    def test_invalid_padding_raises(self) -> None:
        """padding > 1.0 should raise BlitzIDError."""
        detector = object.__new__(FaceDetectorDNN)
        detector.logger = logging.getLogger("test")
        # Mock net so extract_faces won't crash if validation is bypassed
        detector.net = MagicMock()
        detector._cache = None
        detector.multi_scale = False
        detector.scales = (1.0,)
        detector.confidence_threshold = 0.5
        detector.min_face_size = (50, 50)
        detector.nms_threshold = 0.3
        with pytest.raises(BlitzIDError, match="padding"):
            detector.extract_faces(SYNTHETIC_IMAGE, padding=2.0)


# ── Behavioral tests (require model) ──────────────────────────


@pytest.fixture(scope="module")
def detector() -> FaceDetectorDNN:
    """Shared detector instance for the module."""
    return FaceDetectorDNN(
        confidence_threshold=0.5,
        enable_cache=True,
        max_cache_size=16,
    )


class TestCacheBehavior:
    def test_cache_disabled_default(self) -> None:
        det = FaceDetectorDNN(enable_cache=False)
        assert det.get_cache_size() == 0

    def test_clear_cache_when_disabled(self) -> None:
        det = FaceDetectorDNN(enable_cache=False)
        det.clear_cache()  # should not raise
        assert det.get_cache_size() == 0

    def test_cache_eviction_behavior(self) -> None:
        det = FaceDetectorDNN(enable_cache=True, max_cache_size=3)
        # Insert 4 different images to trigger eviction
        for i in range(4):
            img = np.zeros((100, 100, 3), dtype=np.uint8)
            img[0, 0] = [i, i, i]  # make each unique
            det.detect_face(img)
        assert det.get_cache_size() <= 3


class TestFactoryPresets:
    def test_create_fast(self) -> None:
        det = FaceDetectorDNN.create_fast_detector()
        assert det.confidence_threshold == 0.7
        assert det.min_face_size == (80, 80)
        assert det.nms_threshold == 0.4

    def test_create_accurate(self) -> None:
        det = FaceDetectorDNN.create_accurate_detector()
        assert det.confidence_threshold == 0.3
        assert det.min_face_size == (20, 20)
        assert det.nms_threshold == 0.2

    def test_create_balanced(self) -> None:
        det = FaceDetectorDNN.create_balanced_detector()
        assert det.confidence_threshold == 0.5
        assert det.min_face_size == (50, 50)
        assert det.nms_threshold == 0.3


class TestExtractFacesCrops:
    def test_crops_are_valid(self, detector: FaceDetectorDNN) -> None:
        extracted = detector.extract_faces(SYNTHETIC_IMAGE)
        assert isinstance(extracted, list)
        for face_img, (x, y, w, h), conf in extracted:
            assert face_img.ndim == 3
            assert face_img.shape[2] == 3
            assert x >= 0
            assert y >= 0
            assert w >= 0
            assert h >= 0
            assert 0.0 <= float(conf) <= 1.0


class TestVisualize:
    def test_shape_matches_input(self, detector: FaceDetectorDNN) -> None:
        result = detector.visualize_detections(SYNTHETIC_IMAGE)
        assert result.shape == SYNTHETIC_IMAGE.shape

    def test_saves_file(self, detector: FaceDetectorDNN) -> None:
        with tempfile.TemporaryDirectory() as td:
            out = Path(td) / "vis.png"
            detector.visualize_detections(SYNTHETIC_IMAGE, output_path=out)
            assert out.exists()
            assert out.stat().st_size > 0


class TestBatchDetection:
    def test_returns_dict_keyed_by_path(self, detector: FaceDetectorDNN) -> None:
        with tempfile.TemporaryDirectory() as td:
            # Create a small valid image
            import cv2

            img_path = Path(td) / "test.png"
            cv2.imwrite(str(img_path), SYNTHETIC_IMAGE)

            results = detector.detect_faces_batch([str(img_path)], show_progress=False)
            assert isinstance(results, dict)
            assert str(img_path) in results


class TestMultiScale:
    def test_runs_without_error(self) -> None:
        det = FaceDetectorDNN(
            confidence_threshold=0.5,
            multi_scale=True,
            scales=(1.0, 1.5),
        )
        faces = det.detect_face(SYNTHETIC_IMAGE)
        assert isinstance(faces, list)


class TestDetectionMetrics:
    def test_all_fields_populated(self, detector: FaceDetectorDNN) -> None:
        metrics = detector.detect_face_with_metrics(SYNTHETIC_IMAGE)
        assert isinstance(metrics, DetectionMetrics)
        assert isinstance(metrics.faces, list)
        assert isinstance(metrics.processing_time, float)
        assert metrics.processing_time >= 0
        assert isinstance(metrics.image_size, tuple)
        assert len(metrics.image_size) == 2
        assert isinstance(metrics.backend, str)
        assert isinstance(metrics.num_faces, int)
        assert isinstance(metrics.cache_hit, bool)
