"""Custom exceptions for the blitzid package."""


class BlitzIDError(Exception):
    """Base exception for all blitzid errors."""


class ModelError(BlitzIDError):
    """Model download or load failure."""


class ImageError(BlitzIDError):
    """Image loading, validation, or processing failure."""


class MRZError(BlitzIDError):
    """MRZ not found, malformed, or failed check-digit validation."""


class FaceVerificationError(BlitzIDError):
    """Face verification failure (no detectable face, missing landmarks)."""
