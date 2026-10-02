"""Passive face anti-spoofing — live face or a photo of one?

Wraps the MiniFASNetV2 (2.7_80x80) silent-face model behind the
package's engine conventions: ModelManager-backed weights with a
pinned digest, an optional shared detector, and plain frozen records
out. The crop and normalization follow the model's reference
implementation exactly (aspect-preserving 2.7x box expansion, 80x80,
raw BGR 0-255 values), and the three outputs are ``[paper, real,
screen]`` — verified against the upstream sample set.

This is an advisory layer against printed photos and screen images. A
video replay of a moving person and deepfake generations are outside
what any model of this size can promise; pair it with the
challenge-response liveness endpoints for the gate that matters.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
from numpy.typing import NDArray

from blitzid._image import ImageInput, load_image
from blitzid._models import ModelManager, default_model_dir
from blitzid.face.detector import FaceDetectorDNN

__all__ = ["AntiSpoofReader", "AntiSpoofResult"]

ANTISPOOF_MODEL_FILENAME = "minifasnet_v2.onnx"
ANTISPOOF_MODEL_URL = (
    "https://raw.githubusercontent.com/QingHeYang/"
    "Silent-Face-Anti-Spoofing-onnx/main/onnx/2.7_80x80_MiniFASNetV2.onnx"
)
ANTISPOOF_MODEL_SHA256 = (
    "0cbe5caec95c31de9d2ef845cb85407d76aecd1b6a2c0e343f7d35306bfbccb8"
)
ANTISPOOF_INPUT_SIZE = (80, 80)
ANTISPOOF_CROP_SCALE = 2.7


@dataclass(frozen=True)
class AntiSpoofResult:
    """Spoof scores for one detected face.

    Attributes:
        bbox: The face's ``(x, y, w, h)`` box in image pixels.
        confidence: The detector's score for this face.
        live_score: Softmax score of the real class, in ``[0, 1]``.
        paper_score: Score of the printed-photo attack class.
        screen_score: Score of the screen-replay attack class.
    """

    bbox: tuple[int, int, int, int]
    confidence: float
    live_score: float
    paper_score: float
    screen_score: float


def _crop_patch(
    img: NDArray[np.uint8], bbox: tuple[int, int, int, int]
) -> NDArray[np.uint8]:
    """Expand the face box 2.7x (aspect preserved) and crop it.

    The box is capped to the image by sliding rather than shrinking,
    the same boundary handling as the model's reference code.
    """
    height, width = img.shape[:2]
    x, y, box_w, box_h = bbox
    scale = min((height - 1) / box_h, min((width - 1) / box_w, ANTISPOOF_CROP_SCALE))
    new_w, new_h = box_w * scale, box_h * scale
    center_x, center_y = box_w / 2 + x, box_h / 2 + y
    x1 = center_x - new_w / 2
    y1 = center_y - new_h / 2
    x2 = center_x + new_w / 2
    y2 = center_y + new_h / 2
    if x1 < 0:
        x2 -= x1
        x1 = 0.0
    if y1 < 0:
        y2 -= y1
        y1 = 0.0
    if x2 > width - 1:
        x1 -= x2 - width + 1
        x2 = float(width - 1)
    if y2 > height - 1:
        y1 -= y2 - height + 1
        y2 = float(height - 1)
    return img[int(y1) : int(y2) + 1, int(x1) : int(x2) + 1]


def to_blob(
    img: NDArray[np.uint8], bbox: tuple[int, int, int, int]
) -> NDArray[np.float32]:
    """Build the model's ``(1, 3, 80, 80)`` blob: BGR, raw 0-255."""
    patch = cv2.resize(_crop_patch(img, bbox), ANTISPOOF_INPUT_SIZE)
    return np.ascontiguousarray(
        np.expand_dims(patch.astype(np.float32).transpose(2, 0, 1), axis=0)
    )


def _softmax(logits: NDArray[np.floating]) -> NDArray[np.float64]:
    scores = np.exp(logits - np.max(logits))
    return np.asarray(scores / scores.sum(), dtype=np.float64)


class AntiSpoofReader:
    """Scores every detected face as live, printed photo, or screen.

    Args:
        detector: The ``FaceDetectorDNN`` supplying face boxes. An
            internal default is constructed when None.
        model_dir: Where the anti-spoof weights live. Defaults to the
            supplied detector's models directory, else the default one.
        log_level: Logging level for the reader's logger.
        allow_downloads: Whether missing weights may be downloaded.

    Raises:
        BlitzIDError: On a bad parameter or weights that will not load.
    """

    def __init__(
        self,
        detector: FaceDetectorDNN | None = None,
        model_dir: Path | None = None,
        log_level: int = logging.INFO,
        allow_downloads: bool = True,
    ) -> None:
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
            filename=ANTISPOOF_MODEL_FILENAME,
            url=ANTISPOOF_MODEL_URL,
            sha256=ANTISPOOF_MODEL_SHA256,
        )
        self.session = self.model_manager.load_session()
        self._input_name = self.session.get_inputs()[0].name

    def read(self, image_input: ImageInput) -> list[AntiSpoofResult]:
        """Score every face in the image.

        Raises:
            ImageError: If the image cannot be loaded.
        """
        img = load_image(image_input, self.logger)
        faces, _, _ = self.detector.detect_from_array(img)
        return [self._score(img, face.bbox, face.confidence) for face in faces]

    def _score(
        self,
        img: NDArray[np.uint8],
        bbox: tuple[int, int, int, int],
        confidence: float,
    ) -> AntiSpoofResult:
        """Run the three-way classifier for one face crop."""
        blob = to_blob(img, bbox)
        logits = self.session.run(None, {self._input_name: blob})[0][0]
        paper, live, screen = _softmax(logits)
        return AntiSpoofResult(
            bbox=bbox,
            confidence=confidence,
            live_score=round(float(live), 4),
            paper_score=round(float(paper), 4),
            screen_score=round(float(screen), 4),
        )
