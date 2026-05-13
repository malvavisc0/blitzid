"""Unit tests for blitzid.detector pure helpers — IoU, NMS, filter, cache, scales, draw.

All tests are pure-function / fast, no model or GPU required.
"""

from __future__ import annotations

import numpy as np
import pytest

from blitzid.detector import (
    _apply_nms,
    _calculate_iou,
    _compute_hash,
    _draw_detections,
    _filter_by_size,
    _LRUCache,
    _normalize_scales,
)
from blitzid.exceptions import BlitzIDError

# ── _calculate_iou ────────────────────────────────────────────


class TestCalculateIoU:
    def test_identical_boxes(self) -> None:
        box = (10, 10, 50, 50)
        assert _calculate_iou(box, box) == pytest.approx(1.0)

    def test_no_overlap(self) -> None:
        box1 = (0, 0, 10, 10)
        box2 = (100, 100, 10, 10)
        assert _calculate_iou(box1, box2) == pytest.approx(0.0)

    def test_partial_overlap(self) -> None:
        box1 = (0, 0, 10, 10)
        box2 = (5, 5, 10, 10)
        # intersection = 5*5 = 25, union = 100+100-25 = 175
        assert _calculate_iou(box1, box2) == pytest.approx(25 / 175)

    def test_zero_area_box(self) -> None:
        box1 = (10, 10, 0, 10)
        box2 = (10, 10, 10, 10)
        assert _calculate_iou(box1, box2) == pytest.approx(0.0)


# ── _apply_nms ────────────────────────────────────────────────


class TestApplyNMS:
    def test_suppresses_overlapping(self) -> None:
        faces = [
            (10, 10, 50, 50, 0.9),
            (12, 12, 50, 50, 0.8),  # high overlap with first
        ]
        result = _apply_nms(faces, threshold=0.3)
        assert len(result) == 1
        assert result[0][4] == 0.9

    def test_keeps_disjoint(self) -> None:
        faces = [
            (0, 0, 20, 20, 0.9),
            (100, 100, 20, 20, 0.8),
        ]
        result = _apply_nms(faces, threshold=0.3)
        assert len(result) == 2


# ── _filter_by_size ───────────────────────────────────────────


class TestFilterBySize:
    def test_drops_small_keeps_large(self) -> None:
        faces = [
            (0, 0, 10, 10, 0.9),  # too small
            (50, 50, 60, 60, 0.8),  # big enough
            (20, 20, 30, 30, 0.7),  # too small
        ]
        result = _filter_by_size(faces, min_size=(50, 50))
        assert len(result) == 1
        assert result[0] == (50, 50, 60, 60, 0.8)

    def test_empty_list(self) -> None:
        result = _filter_by_size([], min_size=(50, 50))
        assert result == []


# ── _LRUCache ─────────────────────────────────────────────────


class TestLRUCache:
    def test_put_get(self) -> None:
        cache = _LRUCache(max_size=10)
        cache.put("key1", [(10, 10, 50, 50, 0.9)])
        result = cache.get("key1")
        assert result is not None
        assert result[0] == (10, 10, 50, 50, 0.9)

    def test_put_overwrites_existing(self) -> None:
        cache = _LRUCache(max_size=10)
        cache.put("key", [(1, 1, 1, 1, 0.5)])
        cache.put("key", [(2, 2, 2, 2, 0.9)])
        result = cache.get("key")
        assert result is not None
        assert result[0] == (2, 2, 2, 2, 0.9)
        assert cache.size() == 1

    def test_eviction(self) -> None:
        cache = _LRUCache(max_size=2)
        cache.put("a", [(1, 1, 1, 1, 0.1)])
        cache.put("b", [(2, 2, 2, 2, 0.2)])
        cache.put("c", [(3, 3, 3, 3, 0.3)])  # should evict "a"
        assert cache.get("a") is None
        assert cache.get("b") is not None
        assert cache.get("c") is not None

    def test_access_refreshes(self) -> None:
        cache = _LRUCache(max_size=2)
        cache.put("a", [(1, 1, 1, 1, 0.1)])
        cache.put("b", [(2, 2, 2, 2, 0.2)])
        # Access "a" to refresh it
        cache.get("a")
        # Now insert "c" — "b" should be evicted (oldest unaccessed)
        cache.put("c", [(3, 3, 3, 3, 0.3)])
        assert cache.get("a") is not None
        assert cache.get("b") is None
        assert cache.get("c") is not None

    def test_clear(self) -> None:
        cache = _LRUCache(max_size=10)
        cache.put("a", [(1, 1, 1, 1, 0.1)])
        cache.put("b", [(2, 2, 2, 2, 0.2)])
        cache.clear()
        assert cache.size() == 0

    def test_get_missing_key(self) -> None:
        cache = _LRUCache(max_size=10)
        assert cache.get("nonexistent") is None


# ── _compute_hash ─────────────────────────────────────────────


class TestComputeHash:
    def test_deterministic(self) -> None:
        arr = np.ones((10, 10, 3), dtype=np.uint8)
        h1 = _compute_hash(arr)
        h2 = _compute_hash(arr)
        assert h1 == h2

    def test_different_data_different_hash(self) -> None:
        a = np.zeros((10, 10, 3), dtype=np.uint8)
        b = np.ones((10, 10, 3), dtype=np.uint8)
        assert _compute_hash(a) != _compute_hash(b)

    def test_same_values_different_dtype(self) -> None:
        a = np.ones((10, 10), dtype=np.uint8)
        b = np.ones((10, 10), dtype=np.float32)
        assert _compute_hash(a) != _compute_hash(b)


# ── _normalize_scales ─────────────────────────────────────────


class TestNormalizeScales:
    def test_includes_1_0_sorted_deduped(self) -> None:
        result = _normalize_scales((2.0, 1.5))
        assert 1.0 in result
        assert result == tuple(sorted(result))

    def test_empty_raises(self) -> None:
        with pytest.raises(BlitzIDError, match="scales"):
            _normalize_scales(())

    def test_below_1_0_raises(self) -> None:
        with pytest.raises(BlitzIDError, match="scales"):
            _normalize_scales((0.5, 1.0))


# ── _draw_detections ──────────────────────────────────────────


class TestDrawDetections:
    def test_returns_copy_same_shape(self) -> None:
        img = np.zeros((200, 200, 3), dtype=np.uint8)
        faces = [(10, 10, 50, 50, 0.9)]
        result = _draw_detections(img, faces)
        assert result.shape == img.shape
        # Original should not be mutated
        np.testing.assert_array_equal(img, np.zeros((200, 200, 3), dtype=np.uint8))

    def test_no_faces(self) -> None:
        img = np.zeros((200, 200, 3), dtype=np.uint8)
        result = _draw_detections(img, [])
        np.testing.assert_array_equal(result, img)
