"""Tests for blitzid.deepface — _as_bgr, sorting, mock-based.

DeepFace tests are skipped if the deepface package is not installed.
"""

from __future__ import annotations

import logging
from unittest.mock import MagicMock

import numpy as np
import pytest

from blitzid.deepface import FaceDetectorDeepFace

# ── _as_bgr (static, no model needed) ────────────────────────


class TestAsBGR:
    def test_float_0_1_to_uint8(self) -> None:
        """Float array in [0, 1] range → scaled to uint8."""
        arr = np.random.rand(50, 50, 3).astype(np.float32)
        result = FaceDetectorDeepFace._as_bgr(arr)
        assert result.dtype == np.uint8
        assert result.shape == (50, 50, 3)

    def test_float_0_255_no_double_multiply(self) -> None:
        """Float array in [0, 255] range → uint8 without double scaling."""
        arr = np.full((50, 50, 3), 200.0, dtype=np.float32)
        result = FaceDetectorDeepFace._as_bgr(arr)
        assert result.dtype == np.uint8
        # Values should be near 200, not clamped to 255 from 200*255
        assert np.all(result >= 190)

    def test_non_array_returned_unchanged(self) -> None:
        """Non-ndarray input is returned as-is."""
        not_array = "hello"
        result = FaceDetectorDeepFace._as_bgr(not_array)  # type: ignore[arg-type]
        assert result is not_array

    def test_uint8_rgb_converted_to_bgr(self) -> None:
        """uint8 RGB → BGR conversion."""
        arr = np.zeros((50, 50, 3), dtype=np.uint8)
        arr[:, :, 0] = 255  # R channel full
        arr[:, :, 1] = 0
        arr[:, :, 2] = 0
        result = FaceDetectorDeepFace._as_bgr(arr)
        # In BGR output, the R channel (was index 0 in RGB) should now be index 2
        assert result[0, 0, 2] == 255
        assert result[0, 0, 0] == 0


# ── _results_to_faces (mock-based) ────────────────────────────


class TestResultsToFaces:
    @pytest.fixture(autouse=True)
    def _check_deepface(self) -> None:
        pytest.importorskip("deepface")

    def test_sorted_desc_by_confidence(self) -> None:
        detector = FaceDetectorDeepFace.__new__(FaceDetectorDeepFace)
        results = [
            {"facial_area": {"x": 0, "y": 0, "w": 10, "h": 10}, "confidence": 0.5},
            {"facial_area": {"x": 20, "y": 20, "w": 10, "h": 10}, "confidence": 0.9},
            {"facial_area": {"x": 40, "y": 40, "w": 10, "h": 10}, "confidence": 0.7},
        ]
        faces = detector._results_to_faces(results)
        confidences = [f[4] for f in faces]
        assert confidences == sorted(confidences, reverse=True)


# ── detect_face sorting (mock-based) ──────────────────────────


class TestDetectFaceSorted:
    @pytest.fixture(autouse=True)
    def _check_deepface(self) -> None:
        pytest.importorskip("deepface")

    @staticmethod
    def _make_mock_detector(
        fake_results: list[dict],
    ) -> FaceDetectorDeepFace:
        """Build a detector with mocked internals (no real model)."""
        det = FaceDetectorDeepFace.__new__(FaceDetectorDeepFace)
        det.logger = logging.getLogger("test")
        det._load_and_convert_rgb = MagicMock(
            return_value=(
                np.zeros((200, 200, 3), dtype=np.uint8),
                np.zeros((200, 200, 3), dtype=np.uint8),
            )
        )
        det._deepface = MagicMock()
        det._deepface.extract_faces.return_value = fake_results
        det.detector_backend = "mock"
        det.enforce_detection = False
        det.align = False
        return det

    def test_highest_confidence_first(self) -> None:
        fake_results = [
            {
                "face": np.zeros((50, 50, 3), dtype=np.uint8),
                "facial_area": {"x": 0, "y": 0, "w": 50, "h": 50},
                "confidence": 0.3,
            },
            {
                "face": np.zeros((50, 50, 3), dtype=np.uint8),
                "facial_area": {"x": 100, "y": 100, "w": 50, "h": 50},
                "confidence": 0.95,
            },
        ]
        det = self._make_mock_detector(fake_results)
        faces = det.detect_face(np.zeros((200, 200, 3), dtype=np.uint8))
        assert len(faces) == 2
        assert faces[0][4] >= faces[1][4]

    def test_detect_face_with_metrics(self) -> None:
        fake_results = [
            {
                "face": np.zeros((50, 50, 3), dtype=np.uint8),
                "facial_area": {"x": 10, "y": 10, "w": 40, "h": 40},
                "confidence": 0.85,
            },
        ]
        det = self._make_mock_detector(fake_results)
        metrics = det.detect_face_with_metrics(np.zeros((200, 200, 3), dtype=np.uint8))
        assert metrics.num_faces == 1
        assert metrics.processing_time >= 0
        assert metrics.image_size == (200, 200)
        assert "DeepFace" in metrics.backend
        assert metrics.cache_hit is False
