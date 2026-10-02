"""Tests for face attributes (age group, gender, race).

Pure tests run without weights; the reader test needs the FairFace
weights (downloaded on first use) and the SCRFD detector.
"""

from __future__ import annotations

import numpy as np
import pytest

from blitzid import FaceAttributeReader, FaceAttributes
from blitzid.face._attributes import (
    AGE_GROUPS,
    ATTRIBUTE_INPUT_SIZE,
    GENDERS,
    RACES,
    _crop_face,
    to_blob,
)


class TestToBlob:
    def test_blob_shape_and_dtype(self) -> None:
        img = np.zeros((120, 160, 3), dtype=np.uint8)
        blob = to_blob(img, (10, 10, 50, 50))
        assert blob.shape == (1, 3, ATTRIBUTE_INPUT_SIZE[1], ATTRIBUTE_INPUT_SIZE[0])
        assert blob.dtype == np.float32

    def test_black_pixel_normalization(self) -> None:
        img = np.zeros((120, 160, 3), dtype=np.uint8)
        blob = to_blob(img, (0, 0, 160, 120))
        # RGB(0,0,0) → (0 - mean) / std per channel.
        assert blob[0, 0, 0, 0] == pytest.approx(-0.485 / 0.229, abs=1e-5)
        assert blob[0, 1, 0, 0] == pytest.approx(-0.456 / 0.224, abs=1e-5)
        assert blob[0, 2, 0, 0] == pytest.approx(-0.406 / 0.225, abs=1e-5)

    def test_margin_crop_is_clamped_to_the_image(self) -> None:
        img = np.zeros((60, 80, 3), dtype=np.uint8)
        crop = _crop_face(img, (70, 50, 20, 20))
        assert crop.shape == (15, 15, 3)
        assert _crop_face(img, (0, 0, 80, 60)).shape == (60, 80, 3)


class TestVocabulary:
    def test_label_sets(self) -> None:
        assert len(AGE_GROUPS) == 9
        assert GENDERS == ("Male", "Female")
        assert len(RACES) == 7


class TestReader:
    def test_reads_attributes_for_the_specimen_portrait(self) -> None:
        reader = FaceAttributeReader()
        attributes = reader.read("images/bub_der_personalausweis_kopie.jpg")
        assert attributes, "no face found"
        first = attributes[0]
        assert isinstance(first, FaceAttributes)
        assert first.age in AGE_GROUPS
        assert first.gender in GENDERS
        assert first.race in RACES
        assert first.bbox[2] > 0 and first.bbox[3] > 0

    def test_confidences_are_scores(self) -> None:
        reader = FaceAttributeReader()
        first = reader.read("images/bub_der_personalausweis_kopie.jpg")[0]
        assert 0.0 <= first.confidence <= 1.0
        assert 0.0 <= first.age_confidence <= 1.0
        assert 0.0 <= first.gender_confidence <= 1.0
        assert 0.0 <= first.race_confidence <= 1.0

    def test_no_faces_returns_empty(self) -> None:
        reader = FaceAttributeReader()
        blank = np.full((200, 200, 3), 255, dtype=np.uint8)
        assert reader.read(blank) == []
