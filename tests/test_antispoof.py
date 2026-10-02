"""Tests for passive face anti-spoofing (live vs photo-of-a-face).

Pure tests run without weights; the reader test needs the MiniFASNet
weights (downloaded on first use) and the SCRFD detector.
"""

from __future__ import annotations

import numpy as np

from blitzid import AntiSpoofReader, AntiSpoofResult
from blitzid.face._antispoof import ANTISPOOF_INPUT_SIZE, to_blob


class TestToBlob:
    def test_blob_keeps_raw_bgr_values(self) -> None:
        """The model expects raw 0-255 BGR, no normalization."""
        img = np.full((120, 160, 3), 200, dtype=np.uint8)
        blob = to_blob(img, (10, 10, 50, 50))
        assert blob.shape == (1, 3, ANTISPOOF_INPUT_SIZE[1], ANTISPOOF_INPUT_SIZE[0])
        assert blob.dtype == np.float32
        assert float(blob[0, 0, 0, 0]) == 200.0

    def test_crop_is_clamped_and_keeps_box_aspect(self) -> None:
        img = np.zeros((60, 80, 3), dtype=np.uint8)
        blob = to_blob(img, (70, 50, 20, 10))
        assert blob.shape[2:] == ANTISPOOF_INPUT_SIZE[::-1]


class TestReader:
    def test_scores_the_specimen_portrait(self) -> None:
        reader = AntiSpoofReader()
        scores = reader.read("images/bub_der_personalausweis_kopie.jpg")
        assert scores, "no face found"
        first = scores[0]
        assert isinstance(first, AntiSpoofResult)
        assert 0.0 <= first.live_score <= 1.0
        assert 0.0 <= first.paper_score <= 1.0
        assert 0.0 <= first.screen_score <= 1.0
        assert first.live_score + first.paper_score + first.screen_score > 0.99

    def test_no_faces_returns_empty(self) -> None:
        reader = AntiSpoofReader()
        blank = np.full((200, 200, 3), 255, dtype=np.uint8)
        assert reader.read(blank) == []
