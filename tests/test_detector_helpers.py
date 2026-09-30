"""Unit tests for blitzid.face.detector and blitzid.face._scrfd pure helpers.

Covers SCRFD decode, IoU, NMS, filter, cache, scales, and draw.
All tests are pure-function / fast, no model required.
"""

from __future__ import annotations

import numpy as np
import pytest

from blitzid.exceptions import BlitzIDError, ModelError
from blitzid.face._face import Face, LRUCache, compute_hash, draw_detections
from blitzid.face._scrfd import (
    anchor_centers,
    decode_outputs,
    distance2bbox,
    distance2kps,
    letterbox_image,
    map_detections_to_faces,
    validate_architecture,
)
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


class TestValidateSCRFDArchitecture:
    def test_expected_nine_outputs(self) -> None:
        validate_architecture(9)

    def test_unexpected_output_count_raises(self) -> None:
        with pytest.raises(ModelError, match="Unsupported SCRFD architecture"):
            validate_architecture(6)


class TestAnchorCenters:
    def test_shape_and_layout(self) -> None:
        centers = anchor_centers(2, 2, 8)
        assert centers.shape == (8, 2)
        expected = [
            (0, 0),
            (0, 0),
            (8, 0),
            (8, 0),
            (0, 8),
            (0, 8),
            (8, 8),
            (8, 8),
        ]
        np.testing.assert_allclose(centers, expected)


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


class TestDistance2BoxAndKps:
    def test_bbox_corner_coordinates(self) -> None:
        centers = np.array([[100.0, 100.0]], dtype=np.float32)
        dist = np.array([[10.0, 20.0, 30.0, 40.0]], dtype=np.float32)
        box = distance2bbox(centers, dist)
        np.testing.assert_allclose(box, [[90.0, 80.0, 130.0, 140.0]])

    def test_kps_point_coordinates(self) -> None:
        centers = np.array([[100.0, 100.0]], dtype=np.float32)
        dist = np.array(
            [[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]], dtype=np.float32
        )
        kps = distance2kps(centers, dist)
        assert kps.shape == (1, 5, 2)
        np.testing.assert_allclose(
            kps[0],
            [
                [101.0, 102.0],
                [103.0, 104.0],
                [105.0, 106.0],
                [107.0, 108.0],
                [109.0, 110.0],
            ],
        )


class TestDecodeScrfdOutputs:
    @staticmethod
    def _levels(
        scores: np.ndarray, boxes: np.ndarray, kps: np.ndarray
    ) -> tuple[list[np.ndarray], list[np.ndarray], list[np.ndarray], list[np.ndarray]]:
        zeros_scores = np.zeros((2, 1), dtype=np.float32)
        zeros_boxes = np.zeros((2, 4), dtype=np.float32)
        zeros_kps = np.zeros((2, 10), dtype=np.float32)
        return (
            [scores, zeros_scores, zeros_scores],
            [boxes, zeros_boxes, zeros_boxes],
            [kps, zeros_kps, zeros_kps],
            [
                anchor_centers(1, 1, 8),
                anchor_centers(1, 1, 16),
                anchor_centers(1, 1, 32),
            ],
        )

    def test_threshold_filters_and_scales(self) -> None:
        scores = np.array([[0.9], [0.2]], dtype=np.float32)
        boxes = np.array([[1.0, 2.0, 3.0, 4.0], [1.0, 1.0, 1.0, 1.0]], dtype=np.float32)
        kps = np.zeros((2, 10), dtype=np.float32)
        score_out, box_out, kps_out, anchors = self._levels(scores, boxes, kps)

        detections = decode_outputs(score_out, box_out, kps_out, anchors, threshold=0.5)

        assert len(detections) == 1
        x1, y1, x2, y2, conf, landmarks = detections[0]
        assert (x1, y1, x2, y2) == (-8.0, -16.0, 24.0, 32.0)
        assert conf == pytest.approx(0.9)
        # Zero offsets at the stride-8 anchor center (0, 0).
        assert landmarks == ((0.0, 0.0),) * 5

    def test_empty_when_all_below_threshold(self) -> None:
        scores = np.array([[0.1], [0.2]], dtype=np.float32)
        boxes = np.zeros((2, 4), dtype=np.float32)
        kps = np.zeros((2, 10), dtype=np.float32)
        score_out, box_out, kps_out, anchors = self._levels(scores, boxes, kps)

        detections = decode_outputs(score_out, box_out, kps_out, anchors, threshold=0.5)
        assert detections == []


class TestMapDetectionsToFaces:
    def test_scales_back_and_clips(self) -> None:
        detections = [(10.0, 20.0, 50.0, 80.0, 0.9, ((20.0, 30.0),))]
        faces = map_detections_to_faces(detections, scale=0.5, image_size=(100, 100))
        assert faces == [
            Face(bbox=(20, 40, 80, 60), confidence=0.9, landmarks=((40, 60),))
        ]

    def test_drops_degenerate_boxes(self) -> None:
        detections = [(-5.0, -5.0, -1.0, -1.0, 0.8, ())]
        faces = map_detections_to_faces(detections, scale=1.0, image_size=(100, 100))
        assert faces == []
