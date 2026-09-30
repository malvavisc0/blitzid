"""Tests for blitzid._models — ModelManager and default_model_dir.

Uses mocking to avoid actual downloads.
"""

from __future__ import annotations

import logging
import urllib.error
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from blitzid._models import ModelManager, default_model_dir
from blitzid.exceptions import ModelError

logger = logging.getLogger("test_models")


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


class TestModelManager:
    def test_downloads_disabled_raises(self, tmp_path: Path) -> None:
        """Missing model + allow_downloads=False → ModelError."""
        mgr = ModelManager(tmp_path, logger, allow_downloads=False)
        with pytest.raises(ModelError, match="Missing model file"):
            mgr.ensure_model_exists()

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
        mock_ort.assert_called_once_with(
            str(mgr.model_path), providers=["CPUExecutionProvider"]
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
