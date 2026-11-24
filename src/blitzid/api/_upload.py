"""Shared upload validation and image byte codecs for the API.

One code path for /analyze and /crop (not per-endpoint copies): bytes
are read once, checked against the size cap first, and sniffed by magic
bytes. Analyze accepts images and small PDFs (page-rasterized in the
worker); the image-only validators reject PDFs elsewhere. Image
decodability and dimension bounds are proven with ``cv2.imdecode``
before anything is stored.

Caveat: Starlette spools request bodies over 1 MB to a transient OS
temp file that is deleted when the request ends. The spool threshold
can be raised later if unacceptable.
"""

from __future__ import annotations

import base64

import cv2
import numpy as np
from fastapi import HTTPException

from blitzid._image import MAX_DIMENSION, MIN_DIMENSION
from blitzid.api._pdf import is_pdf_bytes, pdf_page_count
from blitzid.exceptions import ImageError

ACCEPTED_IMAGE_FORMATS = "jpg/png/webp/bmp/tiff"


def _check_upload_size(data: bytes, max_bytes: int) -> None:
    """Reject uploads above the byte cap."""
    if len(data) > max_bytes:
        raise HTTPException(
            413, f"upload exceeds the {max_bytes // (1024 * 1024)} MB limit"
        )


def decode_image_bytes(data: bytes) -> np.ndarray | None:
    """Decode raw image bytes into a BGR array, or None if undecodable."""
    if not data:
        return None
    return cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)


def dims_within_bounds(img: np.ndarray) -> bool:
    """Whether the image dimensions match the library's min/max bounds."""
    height, width = int(img.shape[0]), int(img.shape[1])
    return min(height, width) >= MIN_DIMENSION and max(height, width) <= MAX_DIMENSION


def encode_jpeg_b64(img: np.ndarray) -> str:
    """Encode a BGR array as a base64 JPEG string.

    Raises:
        ImageError: If the JPEG encoder fails.
    """
    try:
        ok, buffer = cv2.imencode(".jpg", img)
    except cv2.error as e:
        raise ImageError(f"failed to encode image as JPEG: {e}") from e
    if not ok:
        raise ImageError("failed to encode image as JPEG")
    return base64.b64encode(buffer.tobytes()).decode("ascii")


def validate_analyze_upload(data: bytes, max_bytes: int) -> None:
    """Validate /analyze upload bytes: images or small PDFs.

    Args:
        data: Raw upload bytes.
        max_bytes: Upload size cap in bytes.

    Raises:
        HTTPException: ``413`` above the size cap, ``400`` for
            undecodable bytes or out-of-bounds image or PDF page
            dimensions.
    """
    if is_pdf_bytes(data):
        validate_pdf_upload(data, max_bytes)
    else:
        validate_image_upload(data, max_bytes)


def validate_pdf_upload(data: bytes, max_bytes: int) -> int:
    """Validate uploaded PDF bytes and return the page count.

    Args:
        data: Raw upload bytes.
        max_bytes: Upload size cap in bytes.

    Returns:
        Number of pages in the PDF.

    Raises:
        HTTPException: ``413`` above the size cap, ``400`` for bytes
            that are not a readable PDF or a PDF outside the page or
            page-size bounds.
    """
    _check_upload_size(data, max_bytes)
    try:
        return pdf_page_count(data)
    except ImageError as e:
        raise HTTPException(400, str(e)) from e


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
            uploads (the image-only paths do not take PDFs), ``400`` for
            bytes no OpenCV decoder accepts or images outside the
            dimension bounds.
    """
    _check_upload_size(data, max_bytes)
    if is_pdf_bytes(data):
        raise HTTPException(
            415,
            "PDF uploads are not supported here; "
            f"accepted image formats: {ACCEPTED_IMAGE_FORMATS}",
        )
    img = decode_image_bytes(data)
    if img is None or not dims_within_bounds(img):
        raise HTTPException(
            400,
            "undecodable or out-of-bounds image bytes; "
            f"accepted image formats: {ACCEPTED_IMAGE_FORMATS} "
            f"({MIN_DIMENSION}-{MAX_DIMENSION} px per side)",
        )
    return img
