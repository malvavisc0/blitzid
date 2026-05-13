"""Unit tests for blitzid._image — image loading and validation.

All tests are pure-function / fast, no model or GPU required.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pytest

from blitzid._image import (
    MIN_DIMENSION,
    _load_from_array,
    _load_from_path,
    _normalize_channels,
    _validate_image,
    load_image,
)
from blitzid.exceptions import ImageError

logger = logging.getLogger("test_image")


# ── _load_from_array ─────────────────────────────────────────


class TestLoadFromArray:
    def test_uint8_array_passes(self) -> None:
        arr = np.zeros((100, 100, 3), dtype=np.uint8)
        result = _load_from_array(arr, logger)
        assert result.dtype == np.uint8
        assert result.shape == (100, 100, 3)

    def test_float32_array_passes(self) -> None:
        arr = np.zeros((100, 100, 3), dtype=np.float32)
        result = _load_from_array(arr, logger)
        assert result.dtype == np.float32

    def test_float64_converts_to_float32(self) -> None:
        arr = np.zeros((100, 100, 3), dtype=np.float64)
        result = _load_from_array(arr, logger)
        assert result.dtype == np.float32

    def test_unsupported_dtype_raises(self) -> None:
        arr = np.zeros((100, 100, 3), dtype=np.int16)
        with pytest.raises(ImageError, match="Unsupported image dtype"):
            _load_from_array(arr, logger)

    def test_empty_array_raises(self) -> None:
        arr = np.array([], dtype=np.uint8)
        with pytest.raises(ImageError, match="empty"):
            _load_from_array(arr, logger)


# ── load_image (type dispatch + happy paths) ──────────────────


class TestLoadImage:
    def test_unsupported_type_raises(self) -> None:
        with pytest.raises(ImageError, match="Unsupported"):
            load_image(123, logger)  # type: ignore[arg-type]

    def test_dict_type_raises(self) -> None:
        with pytest.raises(ImageError, match="Unsupported"):
            load_image({"key": "value"}, logger)  # type: ignore[arg-type]

    def test_valid_array_end_to_end(self) -> None:
        arr = np.zeros((100, 100, 3), dtype=np.uint8)
        result = load_image(arr, logger)
        assert result.shape == (100, 100, 3)
        assert result.dtype == np.uint8

    def test_string_path(self, tmp_path: Path) -> None:
        import cv2

        img_path = str(tmp_path / "test.png")
        cv2.imwrite(img_path, np.zeros((100, 100, 3), dtype=np.uint8))
        result = load_image(img_path, logger)
        assert result.shape == (100, 100, 3)


# ── _validate_image ───────────────────────────────────────────


class TestValidateImage:
    def test_too_small_raises(self) -> None:
        arr = np.zeros((MIN_DIMENSION - 1, MIN_DIMENSION, 3), dtype=np.uint8)
        with pytest.raises(ImageError, match="too small"):
            _validate_image(arr)

    def test_one_dimensional_raises(self) -> None:
        arr = np.zeros((100,), dtype=np.uint8)
        with pytest.raises(ImageError, match="at least 2 dimensions"):
            _validate_image(arr)

    def test_exact_minimum_passes(self) -> None:
        arr = np.zeros((MIN_DIMENSION, MIN_DIMENSION, 3), dtype=np.uint8)
        _validate_image(arr)  # should not raise

    def test_none_raises(self) -> None:
        with pytest.raises(ImageError, match="None"):
            _validate_image(None)  # type: ignore[arg-type]

    def test_empty_2d_raises(self) -> None:
        arr = np.zeros((0, 100), dtype=np.uint8)
        with pytest.raises(ImageError, match="empty"):
            _validate_image(arr)


# ── _normalize_channels ───────────────────────────────────────


class TestNormalizeChannels:
    def test_grayscale_2d_to_bgr(self) -> None:
        gray = np.zeros((100, 100), dtype=np.uint8)
        result = _normalize_channels(gray)
        assert len(result.shape) == 3
        assert result.shape[2] == 3

    def test_single_channel_to_bgr(self) -> None:
        single = np.zeros((100, 100, 1), dtype=np.uint8)
        result = _normalize_channels(single)
        assert len(result.shape) == 3
        assert result.shape[2] == 3

    def test_rgba_to_bgr(self) -> None:
        rgba = np.zeros((100, 100, 4), dtype=np.uint8)
        result = _normalize_channels(rgba)
        assert len(result.shape) == 3
        assert result.shape[2] == 3

    def test_unsupported_channels_raises(self) -> None:
        five_channel = np.zeros((100, 100, 5), dtype=np.uint8)
        with pytest.raises(ImageError, match="Unsupported number of channels"):
            _normalize_channels(five_channel)

    def test_three_channels_unchanged(self) -> None:
        bgr = np.zeros((100, 100, 3), dtype=np.uint8)
        bgr[10, 10] = [1, 2, 3]
        result = _normalize_channels(bgr)
        assert result.shape[2] == 3
        np.testing.assert_array_equal(result[10, 10], [1, 2, 3])


# ── _load_from_path ───────────────────────────────────────────


class TestLoadFromPath:
    def test_not_found_raises(self, tmp_path: Path) -> None:
        with pytest.raises(ImageError, match="not found"):
            _load_from_path(tmp_path / "nonexistent.png")

    def test_empty_file_raises(self, tmp_path: Path) -> None:
        empty = tmp_path / "empty.png"
        empty.write_bytes(b"")
        with pytest.raises(ImageError, match="empty"):
            _load_from_path(empty)

    def test_directory_raises(self, tmp_path: Path) -> None:
        with pytest.raises(ImageError, match="not a file"):
            _load_from_path(tmp_path)

    def test_corrupted_file_raises(self, tmp_path: Path) -> None:
        bad = tmp_path / "bad.png"
        bad.write_bytes(b"not a real image at all")
        with pytest.raises(ImageError, match="Could not read"):
            _load_from_path(bad)


# ── _load_from_pil ────────────────────────────────────────────


class TestLoadFromPil:
    @pytest.fixture(autouse=True)
    def _check_pil(self) -> None:
        pytest.importorskip("PIL")

    def test_rgb_pil_converts_to_bgr(self) -> None:
        from PIL import Image as PILImage

        pil_img = PILImage.new("RGB", (100, 100), color=(255, 0, 0))
        from blitzid._image import _load_from_pil

        result = _load_from_pil(pil_img)
        assert isinstance(result, np.ndarray)
        assert len(result.shape) == 3
        assert result.shape[2] == 3

    def test_grayscale_pil_converts_to_bgr(self) -> None:
        from PIL import Image as PILImage

        pil_img = PILImage.new("L", (100, 100), color=128)
        from blitzid._image import _load_from_pil

        result = _load_from_pil(pil_img)
        assert isinstance(result, np.ndarray)
        assert len(result.shape) == 3
        assert result.shape[2] == 3
