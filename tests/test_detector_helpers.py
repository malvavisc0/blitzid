"""Unit tests for blitzid.face.detector and blitzid.face._yunet helpers.

Covers IoU, NMS, filter, cache, scales, letterboxing, YuNet mapping, and
drawing. All tests are pure-function / fast, no model required.
"""

from __future__ import annotations

import numpy as np
import pytest

from blitzid.exceptions import BlitzIDError
from blitzid.face._face import Face, LRUCache, compute_hash, draw_detections
from blitzid.face._yunet import letterbox_image, map_yunet_faces
from blitzid.face.detector import (
    _apply_nms,
    _calculate_iou,
    _filter_by_size,
    _normalize_scales,
)


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


class TestApplyNMS:
    def test_suppresses_overlapping(self) -> None:
        faces = [
            Face(bbox=(10, 10, 50, 50), confidence=0.9),
            Face(bbox=(12, 12, 50, 50), confidence=0.8),
        ]
        result = _apply_nms(faces, threshold=0.3)
        assert len(result) == 1
        assert result[0].confidence == 0.9

    def test_keeps_disjoint(self) -> None:
        faces = [
            Face(bbox=(0, 0, 20, 20), confidence=0.9),
            Face(bbox=(100, 100, 20, 20), confidence=0.8),
        ]
        result = _apply_nms(faces, threshold=0.3)
        assert len(result) == 2


class TestFilterBySize:
    def test_drops_small_keeps_large(self) -> None:
        faces = [
            Face(bbox=(0, 0, 10, 10), confidence=0.9),
            Face(bbox=(50, 50, 60, 60), confidence=0.8),
            Face(bbox=(20, 20, 30, 30), confidence=0.7),
        ]
        result = _filter_by_size(faces, min_size=(50, 50))
        assert len(result) == 1
        assert result[0] == Face(bbox=(50, 50, 60, 60), confidence=0.8)

    def test_empty_list(self) -> None:
        result = _filter_by_size([], min_size=(50, 50))
        assert result == []


class TestLRUCache:
    def test_put_get(self) -> None:
        cache = LRUCache(max_size=10)
        cache.put("key1", [Face(bbox=(10, 10, 50, 50), confidence=0.9)])
        result = cache.get("key1")
        assert result is not None
        assert result[0] == Face(bbox=(10, 10, 50, 50), confidence=0.9)

    def test_put_overwrites_existing(self) -> None:
        cache = LRUCache(max_size=10)
        cache.put("key", [Face(bbox=(1, 1, 1, 1), confidence=0.5)])
        cache.put("key", [Face(bbox=(2, 2, 2, 2), confidence=0.9)])
        result = cache.get("key")
        assert result is not None
        assert result[0] == Face(bbox=(2, 2, 2, 2), confidence=0.9)
        assert cache.size() == 1

    def test_eviction(self) -> None:
        cache = LRUCache(max_size=2)
        cache.put("a", [Face(bbox=(1, 1, 1, 1), confidence=0.1)])
        cache.put("b", [Face(bbox=(2, 2, 2, 2), confidence=0.2)])
        cache.put("c", [Face(bbox=(3, 3, 3, 3), confidence=0.3)])
        assert cache.get("a") is None
        assert cache.get("b") is not None
        assert cache.get("c") is not None

    def test_access_refreshes(self) -> None:
        cache = LRUCache(max_size=2)
        cache.put("a", [Face(bbox=(1, 1, 1, 1), confidence=0.1)])
        cache.put("b", [Face(bbox=(2, 2, 2, 2), confidence=0.2)])
        cache.get("a")
        cache.put("c", [Face(bbox=(3, 3, 3, 3), confidence=0.3)])
        assert cache.get("a") is not None
        assert cache.get("b") is None
        assert cache.get("c") is not None

    def test_clear(self) -> None:
        cache = LRUCache(max_size=10)
        cache.put("a", [Face(bbox=(1, 1, 1, 1), confidence=0.1)])
        cache.put("b", [Face(bbox=(2, 2, 2, 2), confidence=0.2)])
        cache.clear()
        assert cache.size() == 0

    def test_get_missing_key(self) -> None:
        cache = LRUCache(max_size=10)
        assert cache.get("nonexistent") is None


class TestComputeHash:
    def test_deterministic(self) -> None:
        arr = np.ones((10, 10, 3), dtype=np.uint8)
        assert compute_hash(arr) == compute_hash(arr)

    def test_different_data_different_hash(self) -> None:
        a = np.zeros((10, 10, 3), dtype=np.uint8)
        b = np.ones((10, 10, 3), dtype=np.uint8)
        assert compute_hash(a) != compute_hash(b)

    def test_same_values_different_dtype(self) -> None:
        a = np.ones((10, 10), dtype=np.uint8)
        b = np.ones((10, 10), dtype=np.float32)
        assert compute_hash(a) != compute_hash(b)

    def test_same_bytes_different_shape(self) -> None:
        """Same byte content but different layout must not collide."""
        a = np.zeros((4, 1, 3), dtype=np.uint8)
        a[0, 0, 0] = 9
        b = np.zeros((1, 4, 3), dtype=np.uint8)
        b[0, 0, 0] = 9
        assert compute_hash(a) != compute_hash(b)


class TestNormalizeScales:
    def test_deduped_and_sorted_as_given(self) -> None:
        assert _normalize_scales((2.0, 1.5)) == (1.5, 2.0)

    def test_does_not_inject_1_0(self) -> None:
        """scales=(1.5,) runs only the 1.5 pass — no hidden 1.0 pass."""
        assert _normalize_scales((1.5,)) == (1.5,)

    def test_empty_raises(self) -> None:
        with pytest.raises(BlitzIDError, match="scales"):
            _normalize_scales(())

    def test_below_1_0_raises(self) -> None:
        with pytest.raises(BlitzIDError, match="scales"):
            _normalize_scales((0.5, 1.0))

    @pytest.mark.parametrize("scale", [np.nan, np.inf, -np.inf])
    def test_non_finite_raises(self, scale: float) -> None:
        with pytest.raises(BlitzIDError, match="scales"):
            _normalize_scales((scale,))


class TestDrawDetections:
    def test_returns_copy_same_shape(self) -> None:
        img = np.zeros((200, 200, 3), dtype=np.uint8)
        faces = [(10, 10, 50, 50, 0.9)]
        result = draw_detections(img, faces)
        assert result.shape == img.shape
        np.testing.assert_array_equal(img, np.zeros((200, 200, 3), dtype=np.uint8))

    def test_no_faces(self) -> None:
        img = np.zeros((200, 200, 3), dtype=np.uint8)
        result = draw_detections(img, [])
        np.testing.assert_array_equal(result, img)


class TestLetterboxImage:
    def test_pads_wide_image(self) -> None:
        img = np.zeros((100, 200, 3), dtype=np.uint8)
        img[:, :] = 255
        canvas, scale = letterbox_image(img, (100, 100))
        assert canvas.shape == (100, 100, 3)
        assert scale == 0.5
        assert canvas[:50, :].sum() > 0
        assert canvas[50:, :].sum() == 0

    def test_identity_scale(self) -> None:
        img = np.zeros((640, 640, 3), dtype=np.uint8)
        _, scale = letterbox_image(img, (640, 640))
        assert scale == 1.0


def _yunet_row(
    x: float,
    y: float,
    w: float,
    h: float,
    landmarks: tuple[tuple[float, float], ...],
    score: float,
) -> np.ndarray:
    row = [x, y, w, h]
    for px, py in landmarks:
        row.extend([px, py])
    row.append(score)
    return np.asarray([row], dtype=np.float32)


class TestMapYunetFaces:
    def test_scales_back_and_clips(self) -> None:
        row = _yunet_row(10, 20, 50, 80, ((20, 30),) * 5, 0.9)
        faces = map_yunet_faces(row, scale=0.5, image_size=(100, 100))
        assert faces == [
            Face(
                bbox=(20, 40, 80, 60),
                confidence=pytest.approx(0.9),
                landmarks=((40.0, 60.0),) * 5,
            )
        ]

    def test_landmarks_keep_subpixel_precision(self) -> None:
        row = _yunet_row(10, 20, 50, 80, ((20.4, 30.6),) * 5, 0.9)
        faces = map_yunet_faces(row, scale=0.5, image_size=(100, 100))
        x, y = faces[0].landmarks[0]
        assert x == pytest.approx(40.8)
        assert y == pytest.approx(61.2)

    def test_landmarks_clip_to_image(self) -> None:
        row = _yunet_row(10, 20, 50, 80, ((200, -5),) * 5, 0.9)
        faces = map_yunet_faces(row, scale=1.0, image_size=(100, 80))
        assert faces[0].landmarks == ((99.0, 0.0),) * 5

    def test_drops_degenerate_boxes(self) -> None:
        row = _yunet_row(-5, -5, -1, -1, (), 0.8)
        faces = map_yunet_faces(row, scale=1.0, image_size=(100, 100))
        assert faces == []

    def test_no_faces(self) -> None:
        assert map_yunet_faces(None, scale=1.0, image_size=(100, 100)) == []
