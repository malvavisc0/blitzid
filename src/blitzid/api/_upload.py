"""Shared upload validation and image byte codecs for the API.

One code path for /analyze and /crop (not per-endpoint copies): bytes
are read once, checked against the size cap first, PDF uploads are
sniffed out by magic bytes, and decodability is proven with
``cv2.imdecode`` before anything is stored.

Caveat: Starlette spools request bodies over 1 MB to a transient OS
temp file that is deleted when the request ends. The spool threshold
can be raised later if unacceptable.
"""

from __future__ import annotations

import base64

import cv2
import numpy as np
from fastapi import HTTPException

from .._image import MIN_DIMENSION
from ..exceptions import ImageError

ACCEPTED_IMAGE_FORMATS = "jpg/png/webp/bmp/tiff"
PDF_MAGIC = b"%PDF-"


def decode_image_bytes(data: bytes) -> np.ndarray | None:
    """Decode raw image bytes into a BGR array, or None if undecodable."""
    return cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)


def encode_jpeg_b64(img: np.ndarray) -> str:
    """Encode a BGR array as a base64 JPEG string.

    Raises:
        ImageError: If the JPEG encoder fails.
    """
    ok, buffer = cv2.imencode(".jpg", img)
    if not ok:
        raise ImageError("failed to encode image as JPEG")
    return base64.b64encode(buffer.tobytes()).decode("ascii")


def validate_image_upload(data: bytes, max_bytes: int) -> np.ndarray:
    """Validate uploaded image bytes and return the decoded BGR array.

    Args:
        data: Raw upload bytes.
        max_bytes: Upload size cap in bytes.

    Returns:
        The decoded image array (callers may discard it to only prove
        decodability).

    Raises:
        HTTPException: ``413`` above the size cap, ``415`` for PDF
            uploads (the pipeline is image-only), ``400`` for bytes
            no OpenCV decoder accepts or images below the minimum
            dimension.
    """
    if len(data) > max_bytes:
        raise HTTPException(
            413, f"upload exceeds the {max_bytes // (1024 * 1024)} MB limit"
        )
    if data[: len(PDF_MAGIC)] == PDF_MAGIC:
        raise HTTPException(
            415,
            "PDF uploads are not supported; "
            f"accepted image formats: {ACCEPTED_IMAGE_FORMATS}",
        )
    img = decode_image_bytes(data)
    if img is None or min(img.shape[:2]) < MIN_DIMENSION:
        raise HTTPException(
            400,
            "undecodable or too-small image bytes; "
            f"accepted image formats: {ACCEPTED_IMAGE_FORMATS}",
        )
    return img
