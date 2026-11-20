"""YuNet detection constants and output mapping.

YuNet is the OpenCV Zoo millisecond-level face detector (MIT license).
It runs through OpenCV's CPU DNN backend and returns one row per face:
``x, y, w, h``, five ``(x, y)`` landmarks, and a confidence score.
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
from numpy.typing import NDArray

from blitzid.exceptions import ModelError
from blitzid.face._face import Face

YUNET_MODEL_FILENAME = "face_detection_yunet_2026may.onnx"
YUNET_MODEL_URL = (
    "https://github.com/opencv/opencv_zoo/raw/main/models/"
    "face_detection_yunet/face_detection_yunet_2026may.onnx"
)
YUNET_MODEL_SHA256 = "ebafce4e3c118d6554634be5c27ab333b4c047a9a8c3faf1d7cf93101c22f0f0"
YUNET_TOP_K = 5000
YUNET_NUM_LANDMARKS = 5


def create_detector(
    model_path: Path,
    det_size: tuple[int, int],
    confidence_threshold: float,
    nms_threshold: float,
) -> cv2.FaceDetectorYN:
    """Create the OpenCV CPU YuNet detector."""
    try:
        return cv2.FaceDetectorYN.create(
            str(model_path),
            "",
            det_size,
            confidence_threshold,
            nms_threshold,
            YUNET_TOP_K,
        )
    except cv2.error as exc:
        raise ModelError(f"Failed to load YuNet model {model_path}: {exc}") from exc


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


def map_yunet_faces(
    faces: NDArray[np.float32] | None,
    scale: float,
    image_size: tuple[int, int],
) -> list[Face]:
    """Map YuNet rows back to :class:`Face` records in image coordinates."""
    if faces is None:
        return []

    w, h = image_size
    mapped: list[Face] = []
    for row in faces:
        x, y, bw, bh = (float(value) for value in row[:4])
        bx1 = max(0, min(int(x / scale), w - 1))
        by1 = max(0, min(int(y / scale), h - 1))
        bx2 = max(0, min(int((x + bw) / scale), w))
        by2 = max(0, min(int((y + bh) / scale), h))
        if bx2 <= bx1 or by2 <= by1:
            continue

        landmarks = tuple(
            (
                min(max(float(row[4 + 2 * index]) / scale, 0.0), float(w - 1)),
                min(max(float(row[5 + 2 * index]) / scale, 0.0), float(h - 1)),
            )
            for index in range(YUNET_NUM_LANDMARKS)
        )
        mapped.append(
            Face(
                bbox=(bx1, by1, bx2 - bx1, by2 - by1),
                confidence=float(row[14]),
                landmarks=landmarks,
            )
        )
    return mapped
