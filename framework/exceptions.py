"""Custom exceptions for face detector module."""


class FaceDetectorError(Exception):
    """Base exception for FaceDetector."""


class ModelDownloadError(FaceDetectorError):
    """Failed to download model files."""


class ImageLoadError(FaceDetectorError):
    """Failed to load or read image."""


class ImageProcessingError(FaceDetectorError):
    """Error during image processing."""


class CUDAConfigError(FaceDetectorError):
    """Failed to configure CUDA backend."""


class InvalidParameterError(FaceDetectorError):
    """Invalid parameter combination or value."""


class OptionalDependencyError(FaceDetectorError):
    """Optional dependency is missing or failed to import."""
