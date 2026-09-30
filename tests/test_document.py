"""Tests for blitzid.reading.document — quad detection, warp, and QC.

Synthetic-image tests run without models or network (the synthetic
document factory lives in ``conftest.py``). The face_present tests on
the specimen fixture use the real SCRFD detector (weights download on
first use, same as tests/test_detector.py).
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import pytest

from blitzid import BlitzIDError, DocumentCropper, Face, FaceDetectorDNN

_FIXTURES = Path(__file__).parent.parent / "images"


class _StubDetector:
    """FaceDetectorDNN stand-in returning a fixed detection result."""

    def __init__(self, faces: list[Any]) -> None:
        self._faces = faces

    def detect_face_landmarks(self, image_input: Any) -> list[Any]:
        return self._faces


def _face() -> Face:
    return Face(bbox=(100, 100, 80, 80), confidence=0.9)


class TestQuadDetection:
    def test_document_found_with_matching_aspect(
        self, synthetic_document: Callable[..., np.ndarray]
    ) -> None:
        crop, report = DocumentCropper().crop(synthetic_document())
        assert report.checks["document_found"] == "pass"
        assert report.quad is not None and len(report.quad) == 4
        assert crop is not None
        assert report.width / report.height == pytest.approx(983 / 620, rel=0.02)
        assert report.verdict == "pass"

    def test_uniform_noise_rejected(self) -> None:
        noise = (
            np.random.default_rng(1)
            # same canvas size as the synthetic-document factory
            .integers(0, 255, (820, 1240, 3))
            .astype(np.uint8)
        )
        crop, report = DocumentCropper().crop(noise)
        assert crop is None
        assert report.quad is None
        assert report.checks["document_found"] == "fail"
        assert report.verdict == "reject"

    @pytest.mark.parametrize("rotation", [8.0, -8.0])
    def test_rotation_recovered_by_warp(
        self, synthetic_document: Callable[..., np.ndarray], rotation: float
    ) -> None:
        crop, report = DocumentCropper().crop(synthetic_document(rotation=rotation))
        assert report.checks["document_found"] == "pass"
        assert crop is not None
        assert report.width / report.height == pytest.approx(983 / 620, rel=0.02)

    def test_perspective_skew_recovered_by_warp(
        self, synthetic_document: Callable[..., np.ndarray]
    ) -> None:
        crop, report = DocumentCropper().crop(synthetic_document(skew=0.15))
        assert report.checks["document_found"] == "pass"
        assert crop is not None
        assert report.width / report.height == pytest.approx(983 / 620, rel=0.02)


class TestQualityChecks:
    def test_small_document_fails_resolution(
        self, synthetic_document: Callable[..., np.ndarray]
    ) -> None:
        _crop, report = DocumentCropper().crop(
            synthetic_document(width=520, height=330)
        )
        assert report.checks["document_found"] == "pass"
        assert report.checks["resolution"] == "fail"
        assert report.verdict == "reject"

    def test_dark_document_fails_brightness(
        self, synthetic_document: Callable[..., np.ndarray]
    ) -> None:
        _crop, report = DocumentCropper().crop(synthetic_document(mean=15.0))
        assert report.checks["brightness"] == "fail"
        assert report.verdict == "reject"

    def test_blurred_document_flags_sharpness(
        self, synthetic_document: Callable[..., np.ndarray]
    ) -> None:
        _crop, report = DocumentCropper().crop(synthetic_document(blur=3.0))
        assert report.checks["sharpness"] in {"warn", "fail"}


class TestSideSemantics:
    def test_front_with_face_passes(
        self, synthetic_document: Callable[..., np.ndarray]
    ) -> None:
        cropper = DocumentCropper(detector=_StubDetector([_face()]))  # type: ignore[arg-type]
        _crop, report = cropper.crop(synthetic_document(), side="front")
        assert report.checks["face_present"] == "pass"
        assert report.face_found is True
        assert report.verdict == "pass"

    def test_front_without_face_warns_not_rejects(
        self, synthetic_document: Callable[..., np.ndarray]
    ) -> None:
        cropper = DocumentCropper(detector=_StubDetector([]))  # type: ignore[arg-type]
        _crop, report = cropper.crop(synthetic_document(), side="front")
        assert report.checks["face_present"] == "warn"
        assert report.face_found is False
        assert report.verdict == "warn"

    def test_back_face_not_expected(
        self, synthetic_document: Callable[..., np.ndarray]
    ) -> None:
        cropper = DocumentCropper(detector=_StubDetector([]))  # type: ignore[arg-type]
        _crop, report = cropper.crop(synthetic_document(), side="back")
        assert report.checks["face_present"] == "n/a"
        assert report.face_found is None
        assert report.verdict == "pass"

    def test_unknown_reports_face_found_as_information(
        self, synthetic_document: Callable[..., np.ndarray]
    ) -> None:
        cropper = DocumentCropper(detector=_StubDetector([_face()]))  # type: ignore[arg-type]
        _crop, report = cropper.crop(synthetic_document(), side="unknown")
        assert report.checks["face_present"] == "n/a"
        assert report.face_found is True

    def test_no_detector_reports_na(
        self, synthetic_document: Callable[..., np.ndarray]
    ) -> None:
        _crop, report = DocumentCropper().crop(synthetic_document(), side="front")
        assert report.checks["face_present"] == "n/a"
        assert report.face_found is None

    def test_invalid_side_raises(
        self, synthetic_document: Callable[..., np.ndarray]
    ) -> None:
        with pytest.raises(BlitzIDError, match="side"):
            DocumentCropper().crop(synthetic_document(), side="left")


@pytest.fixture(scope="module")
def detector() -> FaceDetectorDNN:
    return FaceDetectorDNN(enable_cache=False, log_level=30)


class TestSpecimenFixture:
    def test_face_present_passes_on_specimen(self, detector: FaceDetectorDNN) -> None:
        image = cv2.imread(str(_FIXTURES / "nl_td1_id_specimen.jpg"))
        crop, report = DocumentCropper(detector=detector).crop(image, side="front")
        assert crop is not None
        assert report.checks["face_present"] == "pass"
        assert report.face_found is True

    def test_face_maps_back_inside_quad(self, detector: FaceDetectorDNN) -> None:
        image = cv2.imread(str(_FIXTURES / "nl_td1_id_specimen.jpg"))
        crop, report = DocumentCropper(detector=detector).crop(image, side="front")
        assert crop is not None and report.quad is not None
        face = detector.detect_face_landmarks(crop)[0]
        fx, fy = face.bbox[0] + face.bbox[2] / 2, face.bbox[1] + face.bbox[3] / 2
        w, h = report.width - 1, report.height - 1
        src = np.array([[0, 0], [w, 0], [w, h], [0, h]], dtype=np.float32)
        matrix = cv2.getPerspectiveTransform(
            src, np.array(report.quad, dtype=np.float32)
        )
        point = matrix @ np.array([fx, fy, 1.0])
        px, py = point[0] / point[2], point[1] / point[2]
        quad = np.array(report.quad, dtype=np.float32)
        assert cv2.pointPolygonTest(quad, (float(px), float(py)), False) >= 0
