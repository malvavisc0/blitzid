"""Image loading and validation for face detection.

Handles loading from file paths, numpy arrays, and PIL Images.
Normalizes all inputs to 3-channel BGR ``np.ndarray``.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING, TypeGuard, Union

import cv2
import numpy as np

from blitzid.exceptions import ImageError

if TYPE_CHECKING:
    from PIL.Image import Image as PILImageType

ImageInput = Union[Path, str, np.ndarray, "PILImageType"]

MIN_DIMENSION = 10
MAX_DIMENSION = 10000


def _is_pil_image(value: object) -> TypeGuard[PILImageType]:
    """Return whether *value* is a PIL Image (PIL is an optional dependency)."""
    try:
        from PIL.Image import Image as PILImage
    except ImportError:  # pragma: no cover
        return False
    return isinstance(value, PILImage)


def load_image(
    image_input: ImageInput,
    logger: logging.Logger,
) -> np.ndarray:
    """Load and validate image from any source.

    Args:
        image_input: Path, numpy array, or PIL Image.
        logger: Logger instance for debug messages.

    Returns:
        Image as numpy array in BGR format.

    Raises:
        ImageError: If image cannot be loaded or is invalid.
    """
    try:
        if isinstance(image_input, np.ndarray):
            img = _load_from_array(image_input, logger)
        elif _is_pil_image(image_input):
            img = _load_from_pil(image_input)
        elif isinstance(image_input, (Path, str)):
            img = _load_from_path(Path(image_input))
        else:
            raise ImageError(f"Unsupported image_input type: {type(image_input)}")

        _validate_image(img)
        img = _normalize_channels(img)
        return img

    except (OSError, ValueError, TypeError, cv2.error) as e:
        raise ImageError(f"Unexpected error loading image: {e}") from e


def _load_from_path(path: Path) -> np.ndarray:
    """Load image from file path."""
    if not path.exists():
        raise ImageError(f"Image not found: {path}")
    if not path.is_file():
        raise ImageError(f"Path is not a file: {path}")
    if path.stat().st_size == 0:
        raise ImageError(f"Image file is empty: {path}")

    img = cv2.imread(str(path))
    if img is None:
        raise ImageError(
            f"Could not read image: {path}. "
            "File may be corrupted or in an unsupported format."
        )
    return img


def _load_from_array(array: np.ndarray, logger: logging.Logger) -> np.ndarray:
    """Validate numpy array."""
    if array.size == 0:
        raise ImageError("Image is empty (size = 0)")

    if array.dtype == np.float64:
        logger.debug("Converting float64 image array to float32")
        array = array.astype(np.float32)
    elif array.dtype not in (np.uint8, np.float32):
        raise ImageError(
            f"Unsupported image dtype: {array.dtype}. "
            "Expected uint8 or float32/float64."
        )

    return np.ascontiguousarray(array)


def _load_from_pil(pil_image: PILImageType) -> np.ndarray:
    """Convert PIL Image to numpy array (BGR)."""
    img_array = np.array(pil_image)
    if img_array.size == 0:
        raise ImageError("PIL Image is empty")

    if len(img_array.shape) == 3 and img_array.shape[2] == 3:
        img = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
    elif len(img_array.shape) == 2:
        img = cv2.cvtColor(img_array, cv2.COLOR_GRAY2BGR)
    else:
        img = img_array

    return img


def _validate_image(img: np.ndarray) -> None:
    """Validate image dimensions and properties."""
    if img.size == 0:
        raise ImageError("Image is empty (size = 0)")
    if len(img.shape) < 2:
        raise ImageError(
            f"Image must have at least 2 dimensions, got shape {img.shape}"
        )

    h, w = img.shape[:2]
    if h < MIN_DIMENSION or w < MIN_DIMENSION:
        raise ImageError(
            f"Image too small: {w}x{h}. Minimum size is {MIN_DIMENSION}x{MIN_DIMENSION}"
        )
    if h > MAX_DIMENSION or w > MAX_DIMENSION:
        raise ImageError(
            f"Image too large: {w}x{h}. Maximum size is {MAX_DIMENSION}x{MAX_DIMENSION}"
        )


def _normalize_channels(img: np.ndarray) -> np.ndarray:
    """Convert to 3-channel BGR."""
    if len(img.shape) == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    elif len(img.shape) == 3:
        channels = img.shape[2]
        if channels == 1:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        elif channels == 4:
            img = cv2.cvtColor(img, cv2.COLOR_RGBA2BGR)
        elif channels != 3:
            raise ImageError(
                f"Unsupported number of channels: {channels}. "
                "Expected 1, 3, or 4 channels."
            )
    return img


def crop_with_padding(
    img: np.ndarray,
    bbox: tuple[int, int, int, int],
    padding: float,
) -> np.ndarray:
    """Crop an ``(x, y, w, h)`` box with relative padding, clipped to the image.

    Args:
        img: BGR image array.
        bbox: The box as ``(x, y, w, h)`` in image pixels.
        padding: Relative padding on each side (fraction of w/h).

    Returns:
        The cropped image region.
    """
    height, width = img.shape[:2]
    x, y, box_w, box_h = bbox
    pad_w, pad_h = int(box_w * padding), int(box_h * padding)
    x1, y1 = max(0, x - pad_w), max(0, y - pad_h)
    x2 = min(width, x + box_w + pad_w)
    y2 = min(height, y + box_h + pad_h)
    return img[y1:y2, x1:x2]
