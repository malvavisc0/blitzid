"""Image loading and validation for face detection."""

import logging
from pathlib import Path
from typing import TYPE_CHECKING, Union

import cv2
import numpy as np

from .exceptions import ImageLoadError, ImageProcessingError

try:
    from PIL import Image as PIL

    PIL_AVAILABLE = True
except ImportError:  # pragma: no cover
    # Keep PIL optional.
    PIL = None
    PIL_AVAILABLE = False

if TYPE_CHECKING:  # pragma: no cover
    from PIL.Image import Image as PILImageType
else:
    PILImageType = object


class ImageLoader:
    """Handles image loading from various sources with validation.

    Optionally applies a *lightweight* preprocessing step to improve robustness
    under mixed lighting. This is intentionally conservative to avoid harming
    well-exposed images.
    """

    MIN_DIMENSION = 10
    MAX_DIMENSION = 10000

    def __init__(
        self,
        logger: logging.Logger,
        preprocess: bool = False,
        clahe_clip_limit: float = 1.2,
        clahe_tile_grid_size: tuple[int, int] = (8, 8),
        preprocess_min_contrast: float = 35.0,
        preprocess_blend: float = 0.35,
    ):
        """Initialize image loader.

        Args:
            logger: Logger instance
            preprocess: If True, apply basic preprocessing after loading
            clahe_clip_limit: CLAHE clip limit used when `preprocess=True`
            clahe_tile_grid_size: CLAHE tile grid used when `preprocess=True`
            preprocess_min_contrast: Minimum grayscale stddev required to skip preprocessing.
                Lower-contrast images (stddev < threshold) may benefit from CLAHE.
            preprocess_blend: Blend factor (0.0-1.0) applied when preprocessing runs.
                `0.0` keeps the original image; `1.0` uses the fully preprocessed image.
        """
        self.logger = logger
        self.preprocess = preprocess
        self.clahe_clip_limit = clahe_clip_limit
        self.clahe_tile_grid_size = clahe_tile_grid_size
        self.preprocess_min_contrast = float(preprocess_min_contrast)
        self.preprocess_blend = float(preprocess_blend)

    def load(
        self, image_input: Union[Path, str, np.ndarray, "PILImageType"]
    ) -> np.ndarray:
        """
        Load and validate image from any source.

        Args:
            image_input: Path, numpy array, or PIL Image

        Returns:
            Image as numpy array in BGR format

        Raises:
            ImageLoadError: If image cannot be loaded or is invalid
            ImageProcessingError: If image has invalid properties
        """
        img = None

        try:
            if isinstance(image_input, np.ndarray):
                img = self._load_from_array(image_input)
            elif (
                PIL_AVAILABLE
                and PIL is not None
                and isinstance(image_input, PIL.Image)
            ):
                img = self._load_from_pil(image_input)
            elif isinstance(image_input, (Path, str)):
                img = self._load_from_path(Path(image_input))
            else:
                raise ImageLoadError(
                    f"Unsupported image_input type: {type(image_input)}"
                )

            # Validate loaded image
            self._validate_image(img)

            # Normalize to 3-channel BGR
            img = self._normalize_channels(img)

            # Optional lightweight cleanup/preprocessing
            if self.preprocess:
                img = self._preprocess(img)

            return img

        except (OSError, ValueError, TypeError, cv2.error) as e:
            # Wrap common IO/decoding/type errors in our public exception.
            raise ImageLoadError(f"Unexpected error loading image: {e}") from e

    def _load_from_path(self, path: Path) -> np.ndarray:
        """
        Load image from file path.

        Args:
            path: Path to image file

        Returns:
            Image as numpy array

        Raises:
            ImageLoadError: If file cannot be loaded
        """
        if not path.exists():
            raise ImageLoadError(f"Image not found: {path}")
        if not path.is_file():
            raise ImageLoadError(f"Path is not a file: {path}")
        if path.stat().st_size == 0:
            raise ImageLoadError(f"Image file is empty: {path}")

        img = cv2.imread(str(path))
        if img is None:
            raise ImageLoadError(
                f"Could not read image: {path}. "
                "File may be corrupted or in an unsupported format."
            )

        return img

    def _load_from_array(self, array: np.ndarray) -> np.ndarray:
        """
        Validate numpy array.

        Args:
            array: Numpy array

        Returns:
            Validated array

        Raises:
            ImageProcessingError: If array is invalid
        """
        if not isinstance(array, np.ndarray):
            raise ImageProcessingError(
                f"Image must be numpy array, got {type(array)}"
            )

        if array.size == 0:
            raise ImageLoadError("Image is empty (size = 0)")

        if array.dtype == np.float64:
            # OpenCV prefers float32; keep behavior explicit.
            self.logger.debug("Converting float64 image array to float32")
            array = array.astype(np.float32)
        elif array.dtype not in (np.uint8, np.float32):
            raise ImageProcessingError(
                f"Unsupported image dtype: {array.dtype}. Expected uint8 or float32/float64."
            )

        # Ensure OpenCV-friendly memory layout.
        return np.ascontiguousarray(array)

    def _load_from_pil(self, pil_image: "PILImageType") -> np.ndarray:
        """
        Convert PIL Image to numpy array.

        Args:
            pil_image: PIL Image object

        Returns:
            Image as numpy array in BGR format

        Raises:
            ImageLoadError: If PIL image is invalid
        """
        img_array = np.array(pil_image)
        if img_array.size == 0:
            raise ImageLoadError("PIL Image is empty")

        # Convert RGB to BGR for OpenCV
        if len(img_array.shape) == 3 and img_array.shape[2] == 3:
            img = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
        elif len(img_array.shape) == 2:
            # Grayscale image
            img = cv2.cvtColor(img_array, cv2.COLOR_GRAY2BGR)
        else:
            img = img_array

        return img

    def _validate_image(self, img: np.ndarray) -> None:
        """
        Validate image dimensions and properties.

        Args:
            img: Image to validate

        Raises:
            ImageLoadError: If image is None or empty
            ImageProcessingError: If image has invalid dimensions
        """
        if img is None:
            raise ImageLoadError("Failed to load image: result is None")

        if not isinstance(img, np.ndarray):
            raise ImageProcessingError(
                f"Image must be numpy array, got {type(img)}"
            )

        if img.size == 0:
            raise ImageLoadError("Image is empty (size = 0)")

        if len(img.shape) < 2:
            raise ImageProcessingError(
                f"Image must have at least 2 dimensions, got shape {img.shape}"
            )

        # Check image dimensions
        h, w = img.shape[:2]
        if h < self.MIN_DIMENSION or w < self.MIN_DIMENSION:
            raise ImageProcessingError(
                f"Image too small: {w}x{h}. "
                f"Minimum size is {self.MIN_DIMENSION}x{self.MIN_DIMENSION}"
            )

        if h > self.MAX_DIMENSION or w > self.MAX_DIMENSION:
            self.logger.warning(
                "Image very large: %dx%d. Processing may be slow. Consider resizing.",
                w,
                h,
            )

    def _normalize_channels(self, img: np.ndarray) -> np.ndarray:
        """
        Convert to 3-channel BGR.

        Args:
            img: Input image

        Returns:
            3-channel BGR image

        Raises:
            ImageProcessingError: If unsupported number of channels
        """
        # Ensure 3-channel BGR image
        if len(img.shape) == 2:
            # Grayscale - convert to BGR
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        elif len(img.shape) == 3:
            if img.shape[2] == 1:
                # Single channel - convert to BGR
                img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
            elif img.shape[2] == 4:
                # RGBA - convert to BGR
                img = cv2.cvtColor(img, cv2.COLOR_RGBA2BGR)
            elif img.shape[2] != 3:
                raise ImageProcessingError(
                    f"Unsupported number of channels: {img.shape[2]}. "
                    "Expected 1, 3, or 4 channels."
                )

        return img

    def _preprocess(self, img: np.ndarray) -> np.ndarray:
        """Apply a conservative preprocessing step.

        Goals:
        - improve robustness under *low contrast* / mixed lighting
        - avoid harming well-exposed images
        - avoid strong color/brightness shifts in saved visualizations

        Strategy (adaptive + mild):
        - compute grayscale contrast (stddev)
        - if contrast is already good, skip preprocessing entirely
        - otherwise apply a *mild* CLAHE on luminance (LAB L channel)
        - preserve mean luminance (avoid global darkening/brightening)
        - blend the result back with the original image

        Notes:
            This is intentionally conservative; it is not a substitute for a
            better model or multi-scale detection.
        """
        # Expect BGR uint8 after normalization.
        if img.dtype != np.uint8:
            return img

        if len(img.shape) != 3 or img.shape[2] != 3:
            return img

        # Fast decision: skip preprocessing unless the image is low-contrast.
        # Using grayscale stddev as a cheap contrast proxy.
        try:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            contrast = float(gray.std())
        except cv2.error as e:  # pragma: no cover
            self.logger.debug(
                "Preprocess contrast check skipped due to error: %s", e
            )
            return img

        if contrast >= float(self.preprocess_min_contrast):
            return img

        blend = float(self.preprocess_blend)
        if blend <= 0.0:
            return img
        if blend > 1.0:
            blend = 1.0

        try:
            lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
            l, a, b = cv2.split(lab)

            clahe = cv2.createCLAHE(
                clipLimit=float(self.clahe_clip_limit),
                tileGridSize=tuple(self.clahe_tile_grid_size),
            )

            l_mean_before = float(np.mean(l))
            l2 = clahe.apply(l)

            # Preserve mean luminance to avoid global darkening.
            l_mean_after = float(np.mean(l2))
            shift = int(round(l_mean_before - l_mean_after))
            if shift != 0:
                l2 = np.clip(l2.astype(np.int16) + shift, 0, 255).astype(
                    np.uint8
                )

            lab2 = cv2.merge((l2, a, b))
            enhanced = cv2.cvtColor(lab2, cv2.COLOR_LAB2BGR)

            # Blend to keep changes subtle.
            out = cv2.addWeighted(img, 1.0 - blend, enhanced, blend, 0.0)
            return out
        except cv2.error as e:  # pragma: no cover
            self.logger.debug("Preprocess step skipped due to error: %s", e)
            return img
