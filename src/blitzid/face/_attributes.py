"""Face attributes — age group, gender, and race from face crops.

Wraps the FairFace attribute model (one CNN with three heads) behind
the package's engine conventions: ModelManager-backed weights with a
pinned digest, an optional shared detector, and plain frozen records
out. Preprocessing matches the model's reference implementation
(bbox crop with a 25% margin, 224x224, ImageNet normalization).

The labels are the model's own vocabulary: nine age bands, Male/Female,
and seven coarse race groups. They are descriptive metadata, not facts.
In particular the race label is a model opinion over coarse groups and
must never decide anything about a person.
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
from blitzid.exceptions import ModelError
from blitzid.face.detector import FaceDetectorDNN

__all__ = ["AGE_GROUPS", "GENDERS", "RACES", "FaceAttributeReader", "FaceAttributes"]

ATTRIBUTE_MODEL_FILENAME = "fairface.onnx"
ATTRIBUTE_MODEL_URL = (
    "https://github.com/yakhyo/fairface-onnx/releases/download/weights/fairface.onnx"
)
ATTRIBUTE_MODEL_SHA256 = (
    "9c8c47d437cd310538d233f2465f9ed0524cb7fb51882a37f74e8bc22437fdbf"
)
ATTRIBUTE_INPUT_SIZE = (224, 224)

AGE_GROUPS = (
    "0-2",
    "3-9",
    "10-19",
    "20-29",
    "30-39",
    "40-49",
    "50-59",
    "60-69",
    "70+",
)
GENDERS = ("Male", "Female")
RACES = (
    "White",
    "Black",
    "Latino_Hispanic",
    "East Asian",
    "Southeast Asian",
    "Indian",
    "Middle Eastern",
)

_CROP_MARGIN = 0.25
_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


@dataclass(frozen=True)
class FaceAttributes:
    """Age group, gender, and race predicted for one face.

    Attributes:
        bbox: The face's ``(x, y, w, h)`` box in image pixels.
        confidence: The detector's score for this face.
        age: Age band, e.g. ``"20-29"`` (one of :data:`AGE_GROUPS`).
        gender: ``"Male"`` or ``"Female"``.
        race: One of the seven coarse groups in :data:`RACES`.
        age_confidence: Softmax score of the winning age band.
        gender_confidence: Softmax score of the winning gender.
        race_confidence: Softmax score of the winning race group.
    """

    bbox: tuple[int, int, int, int]
    confidence: float
    age: str
    gender: str
    race: str
    age_confidence: float
    gender_confidence: float
    race_confidence: float


def _crop_face(
    img: NDArray[np.uint8], bbox: tuple[int, int, int, int]
) -> NDArray[np.uint8]:
    """Crop one face with a 25% margin, clamped to the image."""
    x, y, w, h = bbox
    x1 = max(0, x - int(w * _CROP_MARGIN))
    y1 = max(0, y - int(h * _CROP_MARGIN))
    x2 = min(img.shape[1], x + w + int(w * _CROP_MARGIN))
    y2 = min(img.shape[0], y + h + int(h * _CROP_MARGIN))
    return img[y1:y2, x1:x2]


def to_blob(
    img: NDArray[np.uint8], bbox: tuple[int, int, int, int]
) -> NDArray[np.float32]:
    """Crop one face (25% margin) and build the model's input blob.

    Args:
        img: BGR image array.
        bbox: The face's ``(x, y, w, h)`` box in image pixels.

    Returns:
        ``(1, 3, 224, 224)`` float32 blob, ImageNet-normalized.
    """
    resized = cv2.resize(_crop_face(img, bbox), ATTRIBUTE_INPUT_SIZE)
    rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    normalized = (rgb - _MEAN) / _STD
    return np.ascontiguousarray(
        np.transpose(normalized, (2, 0, 1))[None], dtype=np.float32
    )


def _softmax(logits: NDArray[np.floating]) -> NDArray[np.float64]:
    """Softmax over one head's logits."""
    scores = np.exp(logits - np.max(logits))
    return np.asarray(scores / scores.sum(), dtype=np.float64)


def _winning(
    labels: tuple[str, ...], logits: NDArray[np.floating]
) -> tuple[str, float]:
    """Return the highest-scoring label and its softmax score."""
    scores = _softmax(logits)
    best = int(np.argmax(scores))
    return labels[best], float(scores[best])


class FaceAttributeReader:
    """Predicts age group, gender, and race for every detected face.

    Args:
        detector: The ``FaceDetectorDNN`` supplying face boxes. An
            internal default is constructed when None.
        model_dir: Where the attribute weights live. Defaults to the
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
            filename=ATTRIBUTE_MODEL_FILENAME,
            url=ATTRIBUTE_MODEL_URL,
            sha256=ATTRIBUTE_MODEL_SHA256,
        )
        self.session = self.model_manager.load_session()
        outputs = self.session.get_outputs()
        if len(outputs) != 3:
            raise ModelError(
                f"Expected 3 heads (race, gender, age), got {len(outputs)}"
            )
        names = [output.name for output in outputs]
        wanted = ("race_output", "gender_output", "age_output")
        self._head_order = (
            [names.index(name) for name in wanted]
            if all(name in names for name in wanted)
            else [0, 1, 2]
        )
        self._input_name = self.session.get_inputs()[0].name

    def read(self, image_input: ImageInput) -> list[FaceAttributes]:
        """Predict attributes for every face in the image.

        Raises:
            ImageError: If the image cannot be loaded.
        """
        img = load_image(image_input, self.logger)
        faces, _, _ = self.detector.detect_from_array(img)
        return [self._predict(img, face.bbox, face.confidence) for face in faces]

    def _predict(
        self,
        img: NDArray[np.uint8],
        bbox: tuple[int, int, int, int],
        confidence: float,
    ) -> FaceAttributes:
        """Run the three heads for one face crop."""
        blob = to_blob(img, bbox)
        results = self.session.run(None, {self._input_name: blob})
        race_logits = results[self._head_order[0]]
        gender_logits = results[self._head_order[1]]
        age_logits = results[self._head_order[2]]
        race, race_score = _winning(RACES, race_logits[0])
        gender, gender_score = _winning(GENDERS, gender_logits[0])
        age, age_score = _winning(AGE_GROUPS, age_logits[0])
        return FaceAttributes(
            bbox=bbox,
            confidence=confidence,
            age=age,
            gender=gender,
            race=race,
            age_confidence=round(age_score, 4),
            gender_confidence=round(gender_score, 4),
            race_confidence=round(race_score, 4),
        )
