"""FaceVerifier — ArcFace-based 1:1 face verification.

Runs the InsightFace ``buffalo_m`` ArcFace recognition ONNX model via an
onnxruntime CPU session. Detected faces are aligned to the canonical
ArcFace template by their five SCRFD landmarks, embedded as
L2-normalized vectors, and compared by cosine similarity against a
decision threshold.
"""

from __future__ import annotations

import logging
import time
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from blitzid._image import ImageInput, load_image
from blitzid._models import ModelManager, default_model_dir
from blitzid.exceptions import BlitzIDError, FaceVerificationError
from blitzid.face._arcface import (
    ARCFACE_MODEL_FILENAME,
    ARCFACE_MODEL_SHA256,
    ARCFACE_MODEL_URL,
    ARCFACE_TEMPLATE,
    align_face,
    normalize_embedding,
    to_blob,
    validate_architecture,
)
from blitzid.face._face import Face, VerificationResult
from blitzid.face.detector import FaceDetectorDNN

__all__ = ["FaceVerifier", "VerificationResult"]

DEFAULT_THRESHOLD = 0.4


def _select_face(faces: list[Face], image_input: ImageInput) -> Face:
    """Pick the highest-confidence face, failing when none was found."""
    if not faces:
        raise FaceVerificationError(f"No face detected in image: {image_input}")
    return max(faces, key=lambda f: f.confidence)


class FaceVerifier:
    """ArcFace face verifier.

    Compares two faces by embedding them with ArcFace and measuring
    cosine similarity. Faces come from the SCRFD detector (an internal
    default or one the caller supplies).
    """

    def __init__(
        self,
        detector: FaceDetectorDNN | None = None,
        threshold: float = DEFAULT_THRESHOLD,
        log_level: int = logging.INFO,
        model_dir: Path | None = None,
        allow_downloads: bool = True,
    ):
        self._validate_threshold(threshold)

        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(log_level)

        self.detector = (
            detector
            if detector is not None
            else FaceDetectorDNN(
                model_dir=model_dir if model_dir is not None else default_model_dir(),
                log_level=log_level,
                allow_downloads=allow_downloads,
            )
        )
        if model_dir is None:
            model_dir = self.detector.model_manager.model_dir
        self.backend = "ONNXRuntime"

        self.model_manager = ModelManager(
            model_dir,
            self.logger,
            allow_downloads=allow_downloads,
            filename=ARCFACE_MODEL_FILENAME,
            url=ARCFACE_MODEL_URL,
            sha256=ARCFACE_MODEL_SHA256,
        )
        self.session = self.model_manager.load_session()

        inputs = self.session.get_inputs()
        outputs = self.session.get_outputs()
        validate_architecture(inputs[0].shape, outputs[0].shape)
        self._input_name = inputs[0].name
        self._output_name = outputs[0].name

        self.threshold = threshold

    @staticmethod
    def _validate_threshold(threshold: float) -> None:
        """Fail fast on thresholds outside the cosine-similarity range."""
        if not (-1.0 <= threshold <= 1.0):
            raise BlitzIDError(
                f"threshold must be between -1.0 and 1.0, got {threshold}"
            )

    def embed(
        self,
        image_input: ImageInput,
        face: Face | None = None,
    ) -> NDArray[np.float32]:
        """Embed a face as an L2-normalized ArcFace vector.

        Args:
            image_input: Image holding the face (path, array, or PIL Image).
            face: A previously detected :class:`Face`; when omitted, the
                highest-confidence face is detected first.

        Returns:
            L2-normalized embedding (``ARCFACE_EMBEDDING_DIM`` values).

        Raises:
            FaceVerificationError: If no face is detected or landmarks are
                missing.
        """
        img = load_image(image_input, self.logger)
        if face is None:
            faces, _, _ = self.detector.detect_from_array(img)
            face = _select_face(faces, image_input)
        return self._embed_face(img, face)

    def verify(
        self,
        image_input1: ImageInput,
        image_input2: ImageInput,
    ) -> VerificationResult:
        """Verify that the two images show the same person.

        Uses the highest-confidence face of each image.

        Args:
            image_input1: First image (path, array, or PIL Image).
            image_input2: Second image (path, array, or PIL Image).

        Returns:
            :class:`VerificationResult` with similarity and verdict.
        """
        start = time.time()
        embedding1 = self.embed(image_input1)
        embedding2 = self.embed(image_input2)
        return self._compare(embedding1, embedding2, start)

    def verify_faces(
        self,
        image_input1: ImageInput,
        face1: Face,
        image_input2: ImageInput,
        face2: Face,
    ) -> VerificationResult:
        """Verify two already-detected faces.

        Skips detection, so callers that ran the detector once reuse its
        results without re-detecting.

        Args:
            image_input1: Image containing *face1*.
            face1: Detected face from that image.
            image_input2: Image containing *face2*.
            face2: Detected face from that image.

        Returns:
            :class:`VerificationResult` with similarity and verdict.
        """
        start = time.time()
        img1 = load_image(image_input1, self.logger)
        img2 = load_image(image_input2, self.logger)
        embedding1 = self._embed_face(img1, face1)
        embedding2 = self._embed_face(img2, face2)
        return self._compare(embedding1, embedding2, start)

    @staticmethod
    def similarity(
        embedding1: NDArray[np.float32],
        embedding2: NDArray[np.float32],
    ) -> float:
        """Cosine similarity between two :meth:`embed` outputs.

        Raises:
            FaceVerificationError: If the embeddings differ in length.
        """
        if embedding1.shape != embedding2.shape:
            raise FaceVerificationError(
                f"Embedding shape mismatch: {embedding1.shape} vs {embedding2.shape}"
            )
        return float(np.dot(embedding1, embedding2))

    def _embed_face(
        self,
        img: np.ndarray,
        face: Face,
    ) -> NDArray[np.float32]:
        """Align and embed one detected face of an already-loaded image."""
        if len(face.landmarks) != len(ARCFACE_TEMPLATE):
            raise FaceVerificationError(
                f"Face has {len(face.landmarks)} landmarks, "
                f"expected {len(ARCFACE_TEMPLATE)}; re-detect with "
                "detect_face_landmarks"
            )
        aligned = align_face(img, face.landmarks)
        output = self.session.run(
            [self._output_name], {self._input_name: to_blob(aligned)}
        )
        return normalize_embedding(output[0])

    def _compare(
        self,
        embedding1: NDArray[np.float32],
        embedding2: NDArray[np.float32],
        start: float,
    ) -> VerificationResult:
        """Build the verification result for two embeddings."""
        similarity = self.similarity(embedding1, embedding2)
        return VerificationResult(
            verified=similarity >= self.threshold,
            similarity=similarity,
            threshold=self.threshold,
            processing_time=time.time() - start,
            backend=self.backend,
        )
