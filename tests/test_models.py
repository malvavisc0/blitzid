"""Tests for blitzid._models — ModelManager and default_model_dir.

Uses mocking to avoid actual downloads.
"""

from __future__ import annotations

import hashlib
import logging
import urllib.error
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from blitzid._models import ModelManager, default_model_dir, session_options
from blitzid.exceptions import ModelError
from blitzid.face._arcface import (
    ARCFACE_MODEL_FILENAME,
    ARCFACE_MODEL_URL,
)

logger = logging.getLogger("test_models")


class _FakeResponse:
    """Minimal ``urlopen`` result: a chunked-read context manager."""

    def __init__(self, payload: bytes) -> None:
        self._payload = payload

    def read(self, size: int = -1) -> bytes:
        if size < 0:
            size = len(self._payload)
        chunk, self._payload = self._payload[:size], self._payload[size:]
        return chunk

    def __enter__(self) -> _FakeResponse:
        return self

    def __exit__(self, *args: object) -> None:
        return None


class TestDefaultModelDir:
    def test_directory_exists(self) -> None:
        result = default_model_dir()
        assert result.exists()
        assert result.is_dir()

    def test_subdir(self) -> None:
        result = default_model_dir("weights")
        assert result.exists()
        assert result.name == "weights"
        assert "models" in str(result)

    def test_env_override(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        monkeypatch.setenv("BLITZID_MODELS_DIR", str(tmp_path))
        result = default_model_dir()
        assert result == tmp_path
        assert result.is_dir()

    def test_env_override_subdir(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        monkeypatch.setenv("BLITZID_MODELS_DIR", str(tmp_path))
        result = default_model_dir("rapidocr")
        assert result == tmp_path / "rapidocr"
        assert result.is_dir()


class TestModelManager:
    def test_downloads_disabled_raises(self, tmp_path: Path) -> None:
        """Missing model + allow_downloads=False → ModelError."""
        mgr = ModelManager(tmp_path, logger, allow_downloads=False)
        with pytest.raises(ModelError, match="Missing model file") as excinfo:
            mgr.ensure_model_exists()
        assert str(mgr.model_path) in str(excinfo.value)

    def test_existing_model_not_downloaded(self, tmp_path: Path) -> None:
        mgr = ModelManager(tmp_path, logger, allow_downloads=True)
        mgr.model_path.write_bytes(b"\x00" * 10)
        with patch.object(mgr, "_download_model") as mock_download:
            mgr.ensure_model_exists()
        mock_download.assert_not_called()

    def test_download_failure_cleanup(self, tmp_path: Path) -> None:
        """Mock urlopen to raise → no .tmp files left behind."""
        mgr = ModelManager(tmp_path, logger, allow_downloads=True)
        with (
            patch(
                "blitzid._models.urllib.request.urlopen",
                side_effect=urllib.error.URLError("network error"),
            ),
            pytest.raises(ModelError, match="Error downloading"),
        ):
            mgr._download_model()

        assert list(tmp_path.glob("*.tmp")) == []
        assert not mgr.model_path.exists()

    def test_load_session_success(self, tmp_path: Path) -> None:
        """Session is created on the explicit CPU provider."""
        mgr = ModelManager(tmp_path, logger)
        mgr.model_path.write_bytes(b"\x00" * 10)
        mock_session = MagicMock()

        with (
            patch(
                "blitzid._models.ort.InferenceSession", return_value=mock_session
            ) as mock_ort,
        ):
            session = mgr.load_session()

        assert session is mock_session
        (path,), kwargs = mock_ort.call_args
        assert path == str(mgr.model_path)
        assert kwargs["providers"] == ["CPUExecutionProvider"]
        options = kwargs["sess_options"]
        assert options.get_session_config_entry("session.intra_op.allow_spinning") == (
            "0"
        )

    def test_session_options_disable_spinning_only(self) -> None:
        """Thread counts stay at onnxruntime's defaults; only spinning is off."""
        options = session_options()
        assert options.intra_op_num_threads == 0
        assert options.inter_op_num_threads == 0
        assert options.get_session_config_entry("session.intra_op.allow_spinning") == (
            "0"
        )

    def test_load_session_failure_raises(self, tmp_path: Path) -> None:
        """A corrupt model file fails fast with ModelError."""
        mgr = ModelManager(tmp_path, logger)
        mgr.model_path.write_bytes(b"\x00" * 10)

        with (
            patch(
                "blitzid._models.ort.InferenceSession",
                side_effect=RuntimeError("bad model"),
            ),
            pytest.raises(ModelError, match="Failed to load"),
        ):
            mgr.load_session()


class TestModelManagerCustomModel:
    def test_custom_filename_and_url(self, tmp_path: Path) -> None:
        mgr = ModelManager(
            tmp_path,
            logger,
            filename="arcface.onnx",
            url="https://example.com/arcface.onnx",
        )
        assert mgr.model_path == tmp_path / "arcface.onnx"
        assert mgr.url == "https://example.com/arcface.onnx"

    def test_scrfd_defaults_without_overrides(self, tmp_path: Path) -> None:
        mgr = ModelManager(tmp_path, logger)
        assert mgr.model_path.name == "scrfd_2.5g.onnx"
        assert "buffalo_m" in mgr.url

    def test_custom_url_is_downloaded(self, tmp_path: Path) -> None:
        """The url override is the URL actually fetched, not MODEL_URL."""
        mgr = ModelManager(
            tmp_path,
            logger,
            filename="arcface.onnx",
            url="https://example.com/arcface.onnx",
            sha256="",
        )
        with patch(
            "blitzid._models.urllib.request.urlopen",
            return_value=_FakeResponse(b"weights"),
        ) as mock_urlopen:
            mgr._download_model()

        request = mock_urlopen.call_args[0][0]
        assert request.full_url == "https://example.com/arcface.onnx"
        assert mgr.model_path.read_bytes() == b"weights"

    def test_arcface_manager_downloads_arcface_url(self, tmp_path: Path) -> None:
        """The FaceVerifier model manager fetches the recognition URL."""
        payload = b"weights"
        mgr = ModelManager(
            tmp_path,
            logger,
            filename=ARCFACE_MODEL_FILENAME,
            url=ARCFACE_MODEL_URL,
            sha256=hashlib.sha256(payload).hexdigest(),
        )
        with patch(
            "blitzid._models.urllib.request.urlopen",
            return_value=_FakeResponse(payload),
        ) as mock_urlopen:
            mgr._download_model()

        request = mock_urlopen.call_args[0][0]
        assert request.full_url == ARCFACE_MODEL_URL
        assert mgr.model_path.name == ARCFACE_MODEL_FILENAME


class TestDigestVerification:
    def test_matching_digest_accepted(self, tmp_path: Path) -> None:
        payload = b"weights"
        digest = hashlib.sha256(payload).hexdigest()
        mgr = ModelManager(
            tmp_path,
            logger,
            filename="arcface.onnx",
            url="https://example.com/arcface.onnx",
            sha256=digest,
        )
        with patch(
            "blitzid._models.urllib.request.urlopen",
            return_value=_FakeResponse(payload),
        ):
            mgr._download_model()

        assert mgr.model_path.read_bytes() == payload

    def test_mismatched_digest_rejected(self, tmp_path: Path) -> None:
        mgr = ModelManager(
            tmp_path,
            logger,
            filename="arcface.onnx",
            url="https://example.com/arcface.onnx",
            sha256="0" * 64,
        )
        with (
            patch(
                "blitzid._models.urllib.request.urlopen",
                return_value=_FakeResponse(b"weights"),
            ),
            pytest.raises(ModelError, match="Digest mismatch"),
        ):
            mgr._download_model()

        assert not mgr.model_path.exists()
        assert list(tmp_path.glob("*.tmp")) == []

    def test_empty_digest_skips_check(self, tmp_path: Path) -> None:
        mgr = ModelManager(
            tmp_path,
            logger,
            filename="arcface.onnx",
            url="https://example.com/arcface.onnx",
            sha256="",
        )
        with patch(
            "blitzid._models.urllib.request.urlopen",
            return_value=_FakeResponse(b"weights"),
        ):
            mgr._download_model()

        assert mgr.model_path.read_bytes() == b"weights"
