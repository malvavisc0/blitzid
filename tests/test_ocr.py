"""Tests for blitzid.reading.ocr — result conversion and reader validation.

Pure-function and stubbed-reader tests run without rapidocr, models,
or network. The engine-backed tests download PP-OCR weights on first
use, so they only run when BLITZID_OCR_INTEGRATION=1 is set.
"""

from __future__ import annotations

import logging
import os
import sys
from pathlib import Path

import numpy as np
import pytest

from blitzid import BlitzIDError, ModelError, OCRText, RapidOCRReader
from blitzid.reading.ocr import _polygon_to_xywh, _to_ocr_texts

SPECIMEN = Path(__file__).parent.parent / "images" / "bub_der_personalausweis_kopie.jpg"
_INTEGRATION = bool(os.environ.get("BLITZID_OCR_INTEGRATION"))


class TestPolygonToXYWH:
    def test_axis_aligned_polygon(self) -> None:
        polygon = np.array([[128, 16], [431, 16], [431, 34], [128, 34]])
        assert _polygon_to_xywh(polygon) == (128, 16, 303, 18)

    def test_rotated_polygon_uses_envelope(self) -> None:
        polygon = np.array([[10, 20], [40, 10], [50, 40], [20, 50]], dtype=np.float64)
        assert _polygon_to_xywh(polygon) == (10, 10, 40, 40)


class TestToOCRTexts:
    def test_converts_and_sorts_by_confidence(self) -> None:
        boxes = np.array(
            [
                [[0, 0], [10, 0], [10, 5], [0, 5]],
                [[100, 0], [120, 0], [120, 8], [100, 8]],
            ]
        )
        texts = _to_ocr_texts(boxes, ("low", "high"), (0.5, 0.99))
        assert texts == [
            OCRText(bbox=(100, 0, 20, 8), text="high", confidence=0.99),
            OCRText(bbox=(0, 0, 10, 5), text="low", confidence=0.5),
        ]

    def test_empty_result(self) -> None:
        assert _to_ocr_texts(None, None, None) == []

    def test_txts_none_with_boxes(self) -> None:
        boxes = np.array([[[0, 0], [1, 0], [1, 1], [0, 1]]])
        assert _to_ocr_texts(boxes, None, None) == []

    def test_scores_cast_to_float(self) -> None:
        boxes = np.array([[[0, 0], [1, 0], [1, 1], [0, 1]]])
        texts = _to_ocr_texts(boxes, ("a",), (np.float32(0.75),))
        assert texts[0].confidence == pytest.approx(0.75)

    def test_mismatched_lengths_raise(self) -> None:
        boxes = np.array([[[0, 0], [1, 0], [1, 1], [0, 1]]])
        with pytest.raises(ModelError, match="mismatched pipeline lengths"):
            _to_ocr_texts(boxes, ("a", "b"), (0.9,))


class TestMissingExtra:
    def test_init_raises_without_rapidocr(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setitem(sys.modules, "rapidocr", None)
        with pytest.raises(BlitzIDError, match=r"blitzid\[ocr\]"):
            RapidOCRReader()


def _stub_reader(engine: object) -> RapidOCRReader:
    """Build a RapidOCRReader around a stub engine, no model download."""
    reader = object.__new__(RapidOCRReader)
    reader.logger = logging.getLogger(__name__)
    reader.model_dir = Path(".")
    reader.engine = engine
    return reader


class TestReadSpecimen:
    @pytest.mark.skipif(
        not _INTEGRATION, reason="set BLITZID_OCR_INTEGRATION=1 to download models"
    )
    def test_reads_text_lines(self) -> None:
        pytest.importorskip("rapidocr")
        texts = RapidOCRReader(log_level=logging.WARNING).read(SPECIMEN)
        assert texts
        for line in texts:
            assert line.text
            assert line.confidence > 0
            x, y, w, h = line.bbox
            assert x >= 0 and y >= 0 and w > 0 and h > 0
        confidences = [line.confidence for line in texts]
        assert confidences == sorted(confidences, reverse=True)

    @pytest.mark.skipif(
        not _INTEGRATION, reason="set BLITZID_OCR_INTEGRATION=1 to download models"
    )
    def test_blank_image_yields_no_text(self) -> None:
        pytest.importorskip("rapidocr")
        reader = RapidOCRReader(log_level=logging.WARNING)
        assert reader.read(np.zeros((480, 640, 3), dtype=np.uint8)) == []

    def test_partial_pipeline_result_raises_model_error(self) -> None:
        pytest.importorskip("rapidocr")
        from rapidocr.ch_ppocr_det.utils import TextDetOutput

        def det_only_engine(img: object) -> object:
            return TextDetOutput(boxes=np.zeros((1, 4, 2)), scores=[0.5])

        reader = _stub_reader(det_only_engine)
        with pytest.raises(ModelError, match="partial"):
            reader.read(SPECIMEN)
