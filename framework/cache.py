"""Detection result caching."""

import hashlib
import logging
from collections import OrderedDict
from typing import List, Optional, Tuple

import numpy as np

from .exceptions import InvalidParameterError


class DetectionCache:
    """Manages caching of detection results."""

    def __init__(
        self,
        max_size: int = 100,
        enabled: bool = False,
        logger: Optional[logging.Logger] = None,
    ):
        """
        Initialize detection cache.

        Args:
            max_size: Maximum number of cached entries
            enabled: Whether caching is enabled
            logger: Optional logger instance
        """
        if max_size < 0:
            raise InvalidParameterError(
                f"max_size must be non-negative, got {max_size}"
            )

        self.max_size = max_size
        self.enabled = enabled
        self.logger = logger or logging.getLogger(__name__)

        # Internal storage is immutable tuples to prevent callers from mutating
        # cached values by accident.
        self._cache: (
            "OrderedDict[str, Tuple[Tuple[int, int, int, int, float], ...]]"
        ) = OrderedDict()

    def get(
        self, key: str
    ) -> Optional[List[Tuple[int, int, int, int, float]]]:
        """
        Get cached result.

        Args:
            key: Cache key (image hash)

        Returns:
            Cached faces list or None if not found
        """
        if not self.enabled:
            return None

        result = self._cache.get(key)
        if result is None:
            return None

        # LRU behavior: mark as recently used.
        self._cache.move_to_end(key, last=True)
        self.logger.debug("Cache hit for key: %s...", key[:8])

        # Return a defensive copy (public API remains a list).
        return list(result)

    def put(
        self, key: str, value: List[Tuple[int, int, int, int, float]]
    ) -> None:
        """
        Cache result with automatic size management.

        Args:
            key: Cache key (image hash)
            value: Faces list to cache
        """
        if not self.enabled:
            return

        # Store as immutable tuples to prevent accidental mutation.
        frozen: Tuple[Tuple[int, int, int, int, float], ...] = tuple(value)
        self._cache[key] = frozen
        self._cache.move_to_end(key, last=True)
        self._evict_oldest()

    def clear(self) -> None:
        """Clear all cached results."""
        self._cache.clear()
        self.logger.debug("Cache cleared")

    def size(self) -> int:
        """
        Get number of cached entries.

        Returns:
            Number of entries in cache
        """
        return len(self._cache)

    @staticmethod
    def compute_hash(img: np.ndarray) -> str:
        """Compute a content hash for an image.

        Notes:
            This hashes the full image buffer (`img.tobytes()`), which can be
            relatively expensive for large images. Caching only pays off when
            the detection pipeline is more expensive than hashing.
        """
        try:
            return hashlib.md5(img.tobytes()).hexdigest()
        except (AttributeError, TypeError, ValueError, BufferError):
            # Best-effort fallback (should be rare): hash based on shape and dtype.
            try:
                return hashlib.md5(
                    f"{img.shape}{img.dtype}".encode()
                ).hexdigest()
            except (AttributeError, TypeError) as e:
                raise InvalidParameterError(
                    "compute_hash expects a numpy.ndarray with shape/dtype"
                ) from e

    def _evict_oldest(self) -> None:
        """Evict least-recently-used entries when the cache exceeds max size."""
        while len(self._cache) > self.max_size:
            self._cache.popitem(last=False)
        self.logger.debug("Cache trimmed to %d entries", len(self._cache))
