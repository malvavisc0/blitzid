"""RapidOCRReader — OCR text reading for ID documents.

Runs RapidOCR (PP-OCR-derived ONNX models) on an onnxruntime CPU
session — the same engine and deployment profile as the SCRFD
detector. Requires the ``ocr`` extra (``pip install blitzid[ocr]``).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

from .._image import ImageInput, load_image
from .._models import default_model_dir
from ..exceptions import BlitzIDError, ModelError

if TYPE_CHECKING:
    from rapidocr import RapidOCR

__all__ = ["OCRText", "RapidOCRReader"]


@dataclass(frozen=True)
class OCRText:
    """One recognized text line.

    Attributes:
        bbox: Axis-aligned (x, y, w, h) box in image pixels.
        text: Recognized text.
        confidence: Recognition score in [0, 1].
    """

    bbox: tuple[int, int, int, int]
    text: str
    confidence: float


def _polygon_to_xywh(polygon: np.ndarray) -> tuple[int, int, int, int]:
    """Convert a (4, 2) corner polygon to an axis-aligned (x, y, w, h) box."""
    x = round(float(polygon[:, 0].min()))
    y = round(float(polygon[:, 1].min()))
    w = round(float(polygon[:, 0].max())) - x
    h = round(float(polygon[:, 1].max())) - y
    return x, y, w, h


def _to_ocr_texts(
    boxes: np.ndarray | None,
    txts: tuple[str, ...] | None,
    scores: tuple[float, ...] | None,
) -> list[OCRText]:
    """Convert RapidOCR output into OCRText records.

    Args:
        boxes: (N, 4, 2) corner polygons, or None when nothing was found.
        txts: Recognized strings, or None.
        scores: Recognition scores, or None.

    Returns:
        OCRText records sorted by confidence, descending.
    """
    if boxes is None or txts is None or scores is None:
        return []

    try:
        records = [
            OCRText(bbox=_polygon_to_xywh(box), text=text, confidence=float(score))
            for box, text, score in zip(boxes, txts, scores, strict=True)
        ]
    except ValueError as e:
        raise ModelError(f"RapidOCR returned mismatched pipeline lengths: {e}") from e
    return sorted(records, key=lambda record: record.confidence, reverse=True)


class RapidOCRReader:
    """OCR text reader backed by RapidOCR (onnxruntime CPU).

    PP-OCR ONNX models download on first use into
    ``user_cache_dir("blitzid")/models/rapidocr``.

    Args:
        model_dir: Directory for the OCR model weights. Defaults to the
            blitzid models dir (``BLITZID_MODELS_DIR`` or the platformdirs
            cache).
        log_level: Logging level for the reader's logger.

    Raises:
        BlitzIDError: If the ``ocr`` extra (rapidocr) is not installed.
        ModelError: If the OCR engine cannot be initialized.
    """

    def __init__(
        self,
        model_dir: Path | None = None,
        log_level: int = logging.INFO,
    ) -> None:
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(log_level)

        if model_dir is None:
            model_dir = default_model_dir("rapidocr")
        self.model_dir = Path(model_dir)

        self.engine: RapidOCR = self._create_engine()

    def read(self, image_input: ImageInput) -> list[OCRText]:
        """Read text lines from an image.

        Args:
            image_input: Path, NumPy array (BGR), or PIL Image.

        Returns:
            OCRText records sorted by confidence, descending. No text
            found yields an empty list.

        Raises:
            ImageError: If the image cannot be loaded or is invalid.
            ModelError: If the engine returns a partial pipeline result
                (e.g. detection-only) instead of a full OCR result.
        """
        from rapidocr.utils.output import RapidOCROutput

        img = load_image(image_input, self.logger)
        result = self.engine(img)
        if not isinstance(result, RapidOCROutput):
            raise ModelError(
                f"RapidOCR returned a {type(result).__name__} (partial "
                "pipeline result) instead of a full OCR result."
            )
        texts = _to_ocr_texts(result.boxes, result.txts, result.scores)
        self.logger.info("%d text line(s) read", len(texts))
        return texts

    def _create_engine(self) -> RapidOCR:
        """Build the RapidOCR engine (lazy import of the ``ocr`` extra)."""
        try:
            from rapidocr import RapidOCR
        except ImportError as e:
            raise BlitzIDError(
                "rapidocr is not installed. "
                "Install the OCR extra: pip install blitzid[ocr]"
            ) from e

        self.logger.info("Initializing RapidOCR (model dir: %s)", self.model_dir)
        try:
            return RapidOCR(params={"Global.model_root_dir": str(self.model_dir)})
        except Exception as e:
            raise ModelError(f"Failed to initialize RapidOCR: {e}") from e
