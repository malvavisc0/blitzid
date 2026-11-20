"""SFace recognition alignment, constants, and embedding normalization.

SFace is the OpenCV Zoo MobileFaceNet recognizer trained with the SFace
loss (Apache 2.0). It uses the canonical five-point 112x112 alignment
template, so the existing landmark order and alignment code carry over
unchanged.
"""

from __future__ import annotations

from collections.abc import Sequence

import cv2
import numpy as np
from numpy.typing import NDArray

from blitzid.exceptions import FaceVerificationError, ModelError

SFACE_MODEL_FILENAME = "sface_2021dec.onnx"
SFACE_MODEL_URL = (
    "https://github.com/opencv/opencv_zoo/raw/main/models/"
    "face_recognition_sface/face_recognition_sface_2021dec.onnx"
)
SFACE_MODEL_SHA256 = "0ba9fbfa01b5270c96627c4ef784da859931e02f04419c829e83484087c34e79"
SFACE_INPUT_SIZE = 112
SFACE_EMBEDDING_DIM = 128
SFACE_COSINE_THRESHOLD = 0.363

SFACE_TEMPLATE: NDArray[np.float32] = np.array(
    [
        [38.2946, 51.6963],
        [73.5318, 51.5014],
        [56.0252, 71.7366],
        [41.5493, 92.3655],
        [70.7299, 92.2041],
    ],
    dtype=np.float32,
)


def similarity_transform(
    src: NDArray[np.float32],
    dst: NDArray[np.float32],
) -> NDArray[np.float64]:
    """Least-squares similarity transform (scale · rotation · translation).

    Equivalent to ``skimage.transform.SimilarityTransform().estimate(src,
    dst)`` without the dependency.

    Args:
        src: Source points, shape ``(N, 2)``.
        dst: Destination points, shape ``(N, 2)``.

    Returns:
        2x3 affine matrix mapping *src* onto *dst*.
    """
    num, dim = src.shape
    src_center = src.mean(axis=0)
    dst_center = dst.mean(axis=0)
    src_d = src - src_center
    dst_d = dst - dst_center

    cross = dst_d.T @ src_d / num
    signs = np.ones(dim)
    signs[-1] = -1.0 if np.linalg.det(cross) < 0 else 1.0
    u, s, vt = np.linalg.svd(cross)
    rotation = u @ np.diag(signs) @ vt

    var_src = (src_d**2).sum() / num
    scale = s @ signs / var_src

    matrix = np.eye(dim + 1, dtype=np.float64)
    matrix[:dim, :dim] = scale * rotation
    matrix[:dim, dim] = dst_center - matrix[:dim, :dim] @ src_center
    return matrix[:dim]


def align_face(
    img: NDArray[np.uint8],
    landmarks: Sequence[tuple[float, float]],
) -> NDArray[np.uint8]:
    """Warp *img* to the canonical 112x112 SFace-aligned crop.

    Args:
        img: BGR image array.
        landmarks: Five subpixel ``(x, y)`` landmark points in YuNet order.

    Returns:
        The aligned 112x112 face crop.

    Raises:
        FaceVerificationError: If the landmark count does not match the
            five-point template.
    """
    if len(landmarks) != len(SFACE_TEMPLATE):
        raise FaceVerificationError(
            f"Expected {len(SFACE_TEMPLATE)} landmarks, got {len(landmarks)}"
        )
    src = np.asarray(landmarks, dtype=np.float32)
    matrix = similarity_transform(src, SFACE_TEMPLATE)
    aligned = cv2.warpAffine(
        img,
        matrix,
        (SFACE_INPUT_SIZE, SFACE_INPUT_SIZE),
        borderValue=0.0,
    )
    return aligned.astype(np.uint8, copy=False)


def normalize_embedding(
    vec: NDArray[np.floating],
) -> NDArray[np.float32]:
    """Flatten a raw model output to an L2-normalized float32 embedding.

    Raises:
        ModelError: If the model produced a non-finite or zero-norm embedding.
    """
    flat = np.asarray(vec, dtype=np.float32).flatten()
    if not np.all(np.isfinite(flat)):
        raise ModelError("Model produced a non-finite embedding")
    norm = np.float32(np.linalg.norm(flat))
    if not np.isfinite(norm):
        raise ModelError("Model produced a non-finite embedding norm")
    if norm == 0.0:
        raise ModelError("Model produced a zero-norm embedding")
    flat /= norm
    return flat
