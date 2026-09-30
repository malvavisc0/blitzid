"""Shared test fixtures for the blitzid test suite."""

from __future__ import annotations

from collections.abc import Callable

import cv2
import numpy as np
import pytest

_CANVAS = (1240, 820)


def _synthetic_document(
    width: int = 983,
    height: int = 620,
    rotation: float = 0.0,
    mean: float = 110.0,
    blur: float = 0.0,
    skew: float = 0.0,
) -> np.ndarray:
    """Draw a noisy document patch onto a light background canvas."""
    rng = np.random.default_rng(7)
    patch = np.clip(rng.normal(mean, 30, (height, width, 3)), 0, 255).astype(np.uint8)
    cv2.putText(patch, "SPECIMEN", (60, height // 2), 0, 2.0, (240, 240, 240), 4)
    cv2.rectangle(
        patch, (width - 250, height - 200), (width - 60, height - 60), (10, 10, 10), 8
    )
    if blur:
        patch = cv2.GaussianBlur(patch, (0, 0), blur)

    center = (_CANVAS[0] / 2, _CANVAS[1] / 2)
    corners = np.array(
        [
            [-width / 2, -height / 2],
            [width / 2, -height / 2],
            [width / 2, height / 2],
            [-width / 2, height / 2],
        ],
        dtype=np.float64,
    )
    corners[2] += [skew * width / 2, skew * height / 2]
    rot = cv2.getRotationMatrix2D((0, 0), rotation, 1.0)
    corners = (rot[:, :2] @ corners.T).T + center
    src = np.array(
        [[0, 0], [width - 1, 0], [width - 1, height - 1], [0, height - 1]],
        dtype=np.float32,
    )
    matrix = cv2.getPerspectiveTransform(src, corners.astype(np.float32))
    return cv2.warpPerspective(patch, matrix, _CANVAS, borderValue=(235, 235, 235))


@pytest.fixture
def synthetic_document() -> Callable[..., np.ndarray]:
    """Factory for synthetic document images (see ``_synthetic_document``)."""
    return _synthetic_document
