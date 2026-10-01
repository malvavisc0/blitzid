"""Tests for FaceVerifier and the ArcFace helpers.

Validation and pure-helper tests are fast (no model load or network).
Behavioural tests require the detector and recognition models to be
available.
"""

from __future__ import annotations

import re
import shutil
from pathlib import Path

import cv2
import numpy as np
import pytest

from blitzid.exceptions import BlitzIDError, FaceVerificationError, ModelError
from blitzid.face._arcface import (
    ARCFACE_EMBEDDING_DIM,
    ARCFACE_INPUT_SIZE,
    ARCFACE_MODEL_FILENAME,
    align_face,
    normalize_embedding,
    similarity_transform,
    to_blob,
    validate_architecture,
)
from blitzid.face._face import Face, VerificationResult
from blitzid.face.detector import FaceDetectorDNN
from blitzid.face.verifier import FaceVerifier

_FIXTURE_DIR = Path(__file__).resolve().parents[1] / "images"
_SPECIMEN = _FIXTURE_DIR / "bub_der_personalausweis_kopie.jpg"
_CONFERENCE = _FIXTURE_DIR / "solvay_conference_1927.jpg"

_LANDMARKS: tuple[tuple[float, float], ...] = (
    (38.29, 51.7),
    (73.53, 51.5),
    (56.03, 71.74),
    (41.55, 92.37),
    (70.73, 92.2),
)


class TestParameterValidation:
    def test_invalid_threshold(self) -> None:
        with pytest.raises(BlitzIDError, match="threshold"):
            FaceVerifier._validate_threshold(1.5)

    def test_invalid_threshold_negative(self) -> None:
        with pytest.raises(BlitzIDError, match="threshold"):
            FaceVerifier._validate_threshold(-1.01)

    def test_boundary_thresholds_valid(self) -> None:
        FaceVerifier._validate_threshold(-1.0)
        FaceVerifier._validate_threshold(1.0)


class TestSimilarityTransform:
    def test_recovers_known_transform(self) -> None:
        src = np.array(
            [[10, 20], [60, 25], [35, 55], [15, 80], [58, 78]], dtype=np.float32
        )
        theta = np.deg2rad(12.0)
        scale = 1.3
        rotation = np.array(
            [[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]]
        )
        dst = (scale * rotation @ src.T).T + np.array([5.0, -3.0])
        matrix = similarity_transform(src, dst.astype(np.float32))
        transformed = src @ matrix[:, :2].T + matrix[:, 2]
        assert np.allclose(transformed, dst, atol=1e-3)

    def test_identity(self) -> None:
        pts = np.array([[0, 0], [10, 0], [0, 10]], dtype=np.float32)
        matrix = similarity_transform(pts, pts)
        assert np.allclose(matrix, [[1, 0, 0], [0, 1, 0]], atol=1e-6)

    def test_reflection_is_not_allowed(self) -> None:
        """A similarity transform cannot mirror; the fit stays proper."""
        src = np.array([[0, 0], [10, 0], [0, 10]], dtype=np.float32)
        dst = np.array([[0, 0], [0, 10], [10, 0]], dtype=np.float32)
        matrix = similarity_transform(src, dst)
        assert np.linalg.det(matrix[:, :2]) > 0


class TestAlignFace:
    def test_output_is_template_sized(self) -> None:
        img = np.zeros((200, 200, 3), dtype=np.uint8)
        aligned = align_face(img, _LANDMARKS)
        assert aligned.shape == (ARCFACE_INPUT_SIZE, ARCFACE_INPUT_SIZE, 3)

    def test_wrong_landmark_count_raises(self) -> None:
        img = np.zeros((200, 200, 3), dtype=np.uint8)
        with pytest.raises(ValueError, match="landmarks"):
            align_face(img, _LANDMARKS[:4])


class TestToBlob:
    def test_shape_and_normalization(self) -> None:
        img = np.full((112, 112, 3), 255, dtype=np.uint8)
        blob = to_blob(img)
        assert blob.shape == (1, 3, ARCFACE_INPUT_SIZE, ARCFACE_INPUT_SIZE)
        assert np.allclose(blob, 1.0)


class TestNormalizeEmbedding:
    def test_unit_norm(self) -> None:
        vec = np.array([[3.0, 4.0]], dtype=np.float32)
        out = normalize_embedding(vec)
        assert out.shape == (2,)
        assert np.isclose(np.linalg.norm(out), 1.0)
        assert np.allclose(out, [0.6, 0.8], atol=1e-6)

    def test_zero_vector_raises(self) -> None:
        with pytest.raises(ModelError, match="zero-norm"):
            normalize_embedding(np.zeros((1, 4), dtype=np.float32))


class TestValidateRecognitionArchitecture:
    def test_valid_layout(self) -> None:
        validate_architecture([None, 3, 112, 112], [1, ARCFACE_EMBEDDING_DIM])

    def test_wrong_input_shape_raises(self) -> None:
        with pytest.raises(ModelError, match="input shape"):
            validate_architecture([None, 3, 224, 224], [1, ARCFACE_EMBEDDING_DIM])

    def test_wrong_output_dim_raises(self) -> None:
        with pytest.raises(ModelError, match="output shape"):
            validate_architecture([None, 3, 112, 112], [1, 256])

    def test_none_shape_raises(self) -> None:
        with pytest.raises(ModelError, match="layout"):
            validate_architecture(None, None)


class TestSimilarity:
    def test_identical_vectors(self) -> None:
        vec = np.array([0.6, 0.8], dtype=np.float32)
        assert FaceVerifier.similarity(vec, vec) == pytest.approx(1.0)

    def test_orthogonal_vectors(self) -> None:
        a = np.array([1.0, 0.0], dtype=np.float32)
        b = np.array([0.0, 1.0], dtype=np.float32)
        assert FaceVerifier.similarity(a, b) == pytest.approx(0.0)

    def test_opposite_vectors(self) -> None:
        vec = np.array([0.6, 0.8], dtype=np.float32)
        assert FaceVerifier.similarity(vec, -vec) == pytest.approx(-1.0)

    def test_shape_mismatch_raises(self) -> None:
        a = np.zeros(512, dtype=np.float32)
        b = np.zeros(256, dtype=np.float32)
        with pytest.raises(FaceVerificationError, match="shape mismatch"):
            FaceVerifier.similarity(a, b)


@pytest.fixture(scope="module")
def verifier() -> FaceVerifier:
    """Shared verifier instance for the module."""
    return FaceVerifier()


@pytest.fixture(scope="module")
def specimen_image() -> np.ndarray:
    return cv2.imread(str(_SPECIMEN))


@pytest.fixture(scope="module")
def conference_image() -> np.ndarray:
    return cv2.imread(str(_CONFERENCE))


def _assert_valid_result(result: VerificationResult, threshold: float) -> None:
    assert isinstance(result, VerificationResult)
    assert isinstance(result.verified, bool)
    assert -1.0 <= result.similarity <= 1.0
    assert result.threshold == threshold
    assert result.processing_time >= 0
    assert result.backend == "ONNXRuntime"
    assert result.verified == (result.similarity >= result.threshold)


class TestEmbed:
    def test_shape_and_norm(
        self, verifier: FaceVerifier, specimen_image: np.ndarray
    ) -> None:
        emb = verifier.embed(specimen_image)
        assert emb.shape == (ARCFACE_EMBEDDING_DIM,)
        assert emb.dtype == np.float32
        assert np.isclose(np.linalg.norm(emb), 1.0)

    def test_deterministic(
        self, verifier: FaceVerifier, specimen_image: np.ndarray
    ) -> None:
        assert np.allclose(verifier.embed(specimen_image), verifier.embed(_SPECIMEN))

    def test_no_face_raises(self, verifier: FaceVerifier) -> None:
        with pytest.raises(FaceVerificationError, match="No face detected"):
            verifier.embed(np.zeros((480, 640, 3), dtype=np.uint8))

    def test_missing_landmarks_raises(
        self, verifier: FaceVerifier, specimen_image: np.ndarray
    ) -> None:
        face = Face(bbox=(0, 0, 50, 50), confidence=0.9)
        with pytest.raises(FaceVerificationError, match="landmarks"):
            verifier.embed(specimen_image, face)


class TestVerify:
    def test_same_image_is_verified(
        self, verifier: FaceVerifier, specimen_image: np.ndarray
    ) -> None:
        result = verifier.verify(specimen_image, specimen_image)
        _assert_valid_result(result, verifier.threshold)
        assert result.verified
        assert result.similarity > 0.9

    def test_rescaled_same_face_is_verified(
        self, verifier: FaceVerifier, specimen_image: np.ndarray
    ) -> None:
        big = cv2.resize(specimen_image, None, fx=1.5, fy=1.5)
        result = verifier.verify(specimen_image, big)
        _assert_valid_result(result, verifier.threshold)
        assert result.verified

    def test_two_paths_same_face(self, verifier: FaceVerifier) -> None:
        result = verifier.verify(_SPECIMEN, _SPECIMEN)
        assert result.verified
        assert result.similarity == pytest.approx(1.0)

    def test_different_people_not_verified(
        self, verifier: FaceVerifier, conference_image: np.ndarray
    ) -> None:
        faces = verifier.detector.detect_face_landmarks(conference_image)
        assert len(faces) >= 2
        result = verifier.verify_faces(
            conference_image, faces[0], conference_image, faces[1]
        )
        _assert_valid_result(result, verifier.threshold)
        assert not result.verified
        assert result.similarity < 0.25

    def test_best_face_is_used(
        self, verifier: FaceVerifier, conference_image: np.ndarray
    ) -> None:
        """verify() on the same multi-face image matches that face with itself."""
        result = verifier.verify(conference_image, conference_image)
        _assert_valid_result(result, verifier.threshold)
        assert result.verified

    def test_no_face_raises(self, verifier: FaceVerifier) -> None:
        with pytest.raises(FaceVerificationError, match="No face detected"):
            verifier.verify(np.zeros((480, 640, 3), dtype=np.uint8), _SPECIMEN)

    def test_threshold_controls_the_verdict(
        self, verifier: FaceVerifier, conference_image: np.ndarray
    ) -> None:
        """The same comparison flips verdicts when the threshold moves."""
        faces = verifier.detector.detect_face_landmarks(conference_image)
        strict = verifier.verify_faces(
            conference_image, faces[0], conference_image, faces[1]
        )
        lenient = FaceVerifier(detector=verifier.detector, threshold=0.0).verify_faces(
            conference_image, faces[0], conference_image, faces[1]
        )
        assert strict.similarity == pytest.approx(lenient.similarity)
        assert not strict.verified
        assert lenient.verified
        assert lenient.threshold == 0.0


class TestModelDirResolution:
    def test_blitzid_models_dir_honored(
        self,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Path,
        verifier: FaceVerifier,
    ) -> None:
        """A set BLITZID_MODELS_DIR is where the verifier looks for weights."""
        monkeypatch.setenv("BLITZID_MODELS_DIR", str(tmp_path))
        shutil.copy(verifier.detector.model_manager.model_path, tmp_path)
        with pytest.raises(
            ModelError,
            match=re.escape(str(tmp_path / ARCFACE_MODEL_FILENAME)),
        ):
            FaceVerifier(allow_downloads=False)

    def test_supplied_detector_dir_is_reused(
        self,
        tmp_path: Path,
        verifier: FaceVerifier,
    ) -> None:
        """ArcFace weights are looked up next to a supplied detector's."""
        shutil.copy(verifier.detector.model_manager.model_path, tmp_path)
        detector = FaceDetectorDNN(model_dir=tmp_path, allow_downloads=False)
        with pytest.raises(
            ModelError,
            match=re.escape(str(tmp_path / ARCFACE_MODEL_FILENAME)),
        ):
            FaceVerifier(detector=detector, allow_downloads=False)
