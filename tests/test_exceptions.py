"""Tests for exception hierarchy and backward-compat aliases."""

from __future__ import annotations

from blitzid import (
    BlitzIDError,
    FaceDetectorError,
    ImageError,
    ImageLoadError,
    ImageProcessingError,
    InvalidParameterError,
    ModelDownloadError,
    ModelError,
)


class TestExceptionHierarchy:
    def test_model_error_subclasses_blitzid_error(self) -> None:
        assert issubclass(ModelError, BlitzIDError)

    def test_image_error_subclasses_blitzid_error(self) -> None:
        assert issubclass(ImageError, BlitzIDError)

    def test_blitzid_error_subclasses_exception(self) -> None:
        assert issubclass(BlitzIDError, Exception)


class TestBackwardCompatAliases:
    def test_face_detector_error(self) -> None:
        assert FaceDetectorError is BlitzIDError

    def test_model_download_error(self) -> None:
        assert ModelDownloadError is ModelError

    def test_image_load_error(self) -> None:
        assert ImageLoadError is ImageError

    def test_image_processing_error(self) -> None:
        assert ImageProcessingError is ImageError

    def test_invalid_parameter_error(self) -> None:
        assert InvalidParameterError is BlitzIDError
