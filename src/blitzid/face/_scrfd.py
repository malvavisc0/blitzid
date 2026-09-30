"""SCRFD-2.5G preprocessing, output decoding, and architecture validation."""

from __future__ import annotations

from collections.abc import Sequence

import cv2
import numpy as np
from numpy.typing import NDArray

from ..exceptions import ModelError
from ._face import Face

SCRFD_STRIDES: tuple[int, ...] = (8, 16, 32)
SCRFD_NUM_ANCHORS = 2
SCRFD_NUM_KPS = 5
SCRFD_INPUT_MEAN = 127.5
SCRFD_INPUT_STD = 128.0

# (x1, y1, x2, y2, score, landmarks) in letterbox coordinates.
RawDetection = tuple[float, float, float, float, float, tuple[tuple[float, float], ...]]


def to_blob(img: NDArray[np.uint8]) -> NDArray[np.float32]:
    """BGR HWC uint8 → normalized RGB NCHW float32 batch of one."""
    rgb = img[..., ::-1]
    transposed = rgb.transpose(2, 0, 1)[np.newaxis, ...]
    return (transposed.astype(np.float32) - SCRFD_INPUT_MEAN) / SCRFD_INPUT_STD


def anchor_centers(height: int, width: int, stride: int) -> NDArray[np.float32]:
    """Anchor center points for one FPN level, in input-image pixels."""
    grid_x, grid_y = np.meshgrid(
        np.arange(width, dtype=np.float32), np.arange(height, dtype=np.float32)
    )
    centers = np.stack([grid_x, grid_y], axis=-1).reshape(-1, 2) * stride
    return np.repeat(centers, SCRFD_NUM_ANCHORS, axis=0)


def letterbox_image(
    img: NDArray[np.uint8], det_size: tuple[int, int]
) -> tuple[NDArray[np.uint8], float]:
    """Resize *img* keeping aspect ratio into a top-left placed canvas."""
    h, w = img.shape[:2]
    dw, dh = det_size
    scale = min(dw / w, dh / h)
    nw = min(dw, max(1, round(w * scale)))
    nh = min(dh, max(1, round(h * scale)))

    canvas = np.zeros((dh, dw, 3), dtype=img.dtype)
    canvas[:nh, :nw] = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_LINEAR)
    return canvas, scale


def distance2bbox(
    centers: NDArray[np.float32], distance: NDArray[np.float32]
) -> NDArray[np.float32]:
    """Convert (left, top, right, bottom) distances to corner coordinates."""
    x1 = centers[:, 0] - distance[:, 0]
    y1 = centers[:, 1] - distance[:, 1]
    x2 = centers[:, 0] + distance[:, 2]
    y2 = centers[:, 1] + distance[:, 3]
    return np.stack([x1, y1, x2, y2], axis=-1)


def distance2kps(
    centers: NDArray[np.float32], distance: NDArray[np.float32]
) -> NDArray[np.float32]:
    """Convert keypoint offsets to absolute points, shape (N, 5, 2)."""
    return centers[:, None, :] + distance.reshape(-1, SCRFD_NUM_KPS, 2)


def decode_outputs(
    score_outputs: Sequence[NDArray[np.float32]],
    bbox_outputs: Sequence[NDArray[np.float32]],
    kps_outputs: Sequence[NDArray[np.float32]],
    anchors: Sequence[NDArray[np.float32]],
    threshold: float,
) -> list[RawDetection]:
    """Decode per-level SCRFD outputs into letterboxed detections."""
    detections: list[RawDetection] = []

    for scores, bbox_preds, kps_preds, centers, stride in zip(
        score_outputs, bbox_outputs, kps_outputs, anchors, SCRFD_STRIDES, strict=True
    ):
        flat_scores = scores.reshape(-1)
        mask = flat_scores > threshold
        if not mask.any():
            continue

        boxes = distance2bbox(centers[mask], bbox_preds[mask] * stride)
        keypoints = distance2kps(centers[mask], kps_preds[mask] * stride)
        kept_scores = flat_scores[mask]

        for box, score, points in zip(boxes, kept_scores, keypoints, strict=True):
            landmarks = tuple((float(p[0]), float(p[1])) for p in points)
            detections.append(
                (
                    float(box[0]),
                    float(box[1]),
                    float(box[2]),
                    float(box[3]),
                    float(score),
                    landmarks,
                )
            )

    return detections


def map_detections_to_faces(
    detections: list[RawDetection],
    scale: float,
    image_size: tuple[int, int],
) -> list[Face]:
    """Map letterboxed detections back to :class:`Face` records."""
    w, h = image_size
    faces: list[Face] = []

    for x1, y1, x2, y2, conf, landmarks in detections:
        bx1 = max(0, min(int(x1 / scale), w - 1))
        by1 = max(0, min(int(y1 / scale), h - 1))
        bx2 = max(0, min(int(x2 / scale), w))
        by2 = max(0, min(int(y2 / scale), h))

        if bx2 <= bx1 or by2 <= by1:
            continue

        points = tuple(
            (max(0, min(int(px / scale), w - 1)), max(0, min(int(py / scale), h - 1)))
            for px, py in landmarks
        )
        faces.append(
            Face(
                bbox=(bx1, by1, bx2 - bx1, by2 - by1),
                confidence=conf,
                landmarks=points,
            )
        )

    return faces


def validate_architecture(num_outputs: int) -> None:
    """Fail fast when the ONNX export is not the expected SCRFD layout."""
    expected = 3 * len(SCRFD_STRIDES)
    if num_outputs != expected:
        raise ModelError(
            f"Unsupported SCRFD architecture: expected {expected} outputs "
            f"(scores, boxes, keypoints per FPN level), got {num_outputs}"
        )
