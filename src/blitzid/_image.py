"""Image loading and validation for face detection.

Handles loading from file paths, numpy arrays, and PIL Images.
Normalizes all inputs to 3-channel BGR ``np.ndarray``.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING, Union

import cv2
import numpy as np

from .exceptions import ImageError

try:
    from PIL import Image as PIL  # type: ignore[import-not-found]

    PIL_AVAILABLE = True
except ImportError:  # pragma: no cover
    PIL = None
    PIL_AVAILABLE = False

if TYPE_CHECKING:  # pragma: no cover
    from PIL.Image import Image as PILImageType  # type: ignore[import-not-found]
else:
    # Use a unique sentinel - not ``object``, which matches *everything*.
    PILImageType = type("_PILStub", (), {})

ImageInput = Union[Path, str, np.ndarray, "PILImageType"]

MIN_DIMENSION = 10
MAX_DIMENSION = 10000


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
        elif PIL_AVAILABLE and PIL is not None and isinstance(image_input, PIL.Image):
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


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------


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
    if not isinstance(array, np.ndarray):
        raise ImageError(f"Image must be numpy array, got {type(array)}")
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
    if img is None:
        raise ImageError("Failed to load image: result is None")
    if not isinstance(img, np.ndarray):
        raise ImageError(f"Image must be numpy array, got {type(img)}")
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
