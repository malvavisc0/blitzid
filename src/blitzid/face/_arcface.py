"""ArcFace recognition preprocessing, alignment, and architecture validation."""

from __future__ import annotations

from collections.abc import Sequence

import cv2
import numpy as np
from numpy.typing import NDArray

from ..exceptions import ModelError

ARCFACE_MODEL_URL = (
    "https://huggingface.co/immich-app/buffalo_m/resolve/main/recognition/model.onnx"
)
ARCFACE_MODEL_FILENAME = "arcface_buffalo_m.onnx"
ARCFACE_MODEL_SHA256 = (
    "4c06341c33c2ca1f86781dab0e829f88ad5b64be9fba56e56bc9ebdefc619e43"
)

ARCFACE_INPUT_SIZE = 112
ARCFACE_EMBEDDING_DIM = 512
ARCFACE_INPUT_MEAN = 127.5
ARCFACE_INPUT_STD = 127.5

# Reference landmark positions for the canonical 112x112 ArcFace-aligned
# crop. Index *i* pairs with SCRFD keypoint *i* (InsightFace convention).
ARCFACE_TEMPLATE: NDArray[np.float32] = np.array(
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
    dst)`` — the alignment InsightFace uses — without the dependency.

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
    """Warp *img* to the canonical 112x112 ArcFace-aligned crop.

    Args:
        img: BGR image array.
        landmarks: Five subpixel ``(x, y)`` landmark points in SCRFD order.

    Returns:
        The aligned 112x112 face crop.
    """
    if len(landmarks) != len(ARCFACE_TEMPLATE):
        raise ValueError(
            f"Expected {len(ARCFACE_TEMPLATE)} landmarks, got {len(landmarks)}"
        )
    src = np.asarray(landmarks, dtype=np.float32)
    matrix = similarity_transform(src, ARCFACE_TEMPLATE)
    aligned = cv2.warpAffine(
        img,
        matrix,
        (ARCFACE_INPUT_SIZE, ARCFACE_INPUT_SIZE),
        borderValue=0.0,
    )
    return aligned.astype(np.uint8, copy=False)


def to_blob(img: NDArray[np.uint8]) -> NDArray[np.float32]:
    """BGR HWC uint8 → normalized RGB NCHW float32 batch of one."""
    rgb = img[..., ::-1]
    transposed = rgb.transpose(2, 0, 1)[np.newaxis, ...]
    return (transposed.astype(np.float32) - ARCFACE_INPUT_MEAN) / ARCFACE_INPUT_STD


def normalize_embedding(
    vec: NDArray[np.floating],
) -> NDArray[np.float32]:
    """Flatten a raw model output to an L2-normalized float32 embedding.

    Raises:
        ModelError: If the model produced a zero-norm embedding.
    """
    flat = np.asarray(vec, dtype=np.float32).flatten()
    norm = np.float32(np.linalg.norm(flat))
    if norm == 0.0:
        raise ModelError("Model produced a zero-norm embedding")
    flat /= norm
    return flat


def validate_architecture(
    input_shape: Sequence[int | str] | None,
    output_shape: Sequence[int | str] | None,
) -> None:
    """Fail fast when the ONNX export is not the expected ArcFace layout.

    Expects a single RGB 112x112 input and a 512-dimensional embedding
    output.
    """
    if input_shape is None or len(input_shape) != 4:
        raise ModelError(f"Unsupported recognition input layout: {input_shape}")
    if tuple(input_shape[1:]) != (3, ARCFACE_INPUT_SIZE, ARCFACE_INPUT_SIZE):
        raise ModelError(
            f"Unsupported recognition input shape: expected "
            f"(N, 3, {ARCFACE_INPUT_SIZE}, {ARCFACE_INPUT_SIZE}), got {input_shape}"
        )
    if output_shape is None or len(output_shape) != 2:
        raise ModelError(f"Unsupported recognition output layout: {output_shape}")
    if output_shape[1] != ARCFACE_EMBEDDING_DIM:
        raise ModelError(
            f"Unsupported recognition output shape: expected "
            f"(N, {ARCFACE_EMBEDDING_DIM}), got {output_shape}"
        )
