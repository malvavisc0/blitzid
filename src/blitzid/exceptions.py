"""Custom exceptions for the blitzid package."""


class BlitzIDError(Exception):
    """Base exception for all blitzid errors."""


class ModelError(BlitzIDError):
    """Model download or load failure."""


class ImageError(BlitzIDError):
    """Image loading, validation, or processing failure."""
