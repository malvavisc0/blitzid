"""Tests for blitzid._models — ModelManager and default_model_dir.

Uses mocking to avoid actual downloads and CUDA requirements.
"""

from __future__ import annotations

import logging
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from blitzid._models import ModelManager, default_model_dir
from blitzid.exceptions import ModelError

logger = logging.getLogger("test_models")


# ── default_model_dir ─────────────────────────────────────────


class TestDefaultModelDir:
    def test_directory_exists(self) -> None:
        result = default_model_dir()
        assert result.exists()
        assert result.is_dir()

    def test_subdir(self) -> None:
        result = default_model_dir("deepface")
        assert result.exists()
        assert result.name == "deepface"
        assert "models" in str(result)


# ── ModelManager (mock-based) ─────────────────────────────────


class TestModelManager:
    def test_downloads_disabled_raises(self, tmp_path: Path) -> None:
        """Missing model files + allow_downloads=False → ModelError."""
        mgr = ModelManager(tmp_path, logger, allow_downloads=False)
        with pytest.raises(ModelError, match="Missing model files"):
            mgr.ensure_models_exist()

    def test_download_failure_cleanup(self, tmp_path: Path) -> None:
        """Mock urlopen to raise → no .tmp files left behind."""
        import urllib.error

        mgr = ModelManager(tmp_path, logger, allow_downloads=True)
        with (
            patch(
                "blitzid._models.urllib.request.urlopen",
                side_effect=urllib.error.URLError("network error"),
            ),
            pytest.raises(ModelError, match="Error downloading"),
        ):
            mgr._download_models()

        # Verify no .tmp files remain
        tmp_files = list(tmp_path.glob("*.tmp"))
        assert tmp_files == []

    def test_cpu_fallback(self, tmp_path: Path) -> None:
        """Mock no CUDA support → backend == CPU."""
        # Create dummy model files so ensure_models_exist passes
        (tmp_path / "deploy.prototxt").write_text("dummy")
        (tmp_path / "res10_300x300_ssd_iter_140000.caffemodel").write_bytes(
            b"\x00" * 100
        )

        mgr = ModelManager(tmp_path, logger)

        with (
            patch.object(mgr, "_opencv_has_cuda_support", return_value=False),
            patch("cv2.dnn.readNetFromCaffe") as mock_read,
        ):
            mock_net = MagicMock()
            mock_read.return_value = mock_net

            _net, backend = mgr.load_network(use_cuda=True, require_cuda=False)
            assert backend == "CPU"

    def test_require_cuda_raises(self, tmp_path: Path) -> None:
        """require_cuda=True + no CUDA → ModelError."""
        (tmp_path / "deploy.prototxt").write_text("dummy")
        (tmp_path / "res10_300x300_ssd_iter_140000.caffemodel").write_bytes(
            b"\x00" * 100
        )

        mgr = ModelManager(tmp_path, logger)

        with (
            patch.object(mgr, "_opencv_has_cuda_support", return_value=False),
            patch("cv2.dnn.readNetFromCaffe") as mock_read,
            pytest.raises(ModelError, match="CUDA"),
        ):
            mock_read.return_value = MagicMock()
            mgr.load_network(use_cuda=True, require_cuda=True)

    def test_load_network_cuda_disabled(self, tmp_path: Path) -> None:
        """use_cuda=False → always CPU backend."""
        (tmp_path / "deploy.prototxt").write_text("dummy")
        (tmp_path / "res10_300x300_ssd_iter_140000.caffemodel").write_bytes(
            b"\x00" * 100
        )

        mgr = ModelManager(tmp_path, logger)

        with patch("cv2.dnn.readNetFromCaffe") as mock_read:
            mock_read.return_value = MagicMock()
            _net, backend = mgr.load_network(use_cuda=False)
            assert backend == "CPU"


# ── _opencv_has_cuda_support ──────────────────────────────────


class TestOpenCVCudaCheck:
    def test_cuda_yes(self) -> None:
        with patch("cv2.getBuildInformation", return_value="  NVIDIA CUDA: YES\n"):
            assert ModelManager._opencv_has_cuda_support() is True

    def test_cuda_no(self) -> None:
        with patch("cv2.getBuildInformation", return_value="  NVIDIA CUDA: NO\n"):
            assert ModelManager._opencv_has_cuda_support() is False

    def test_no_cuda_line(self) -> None:
        with patch("cv2.getBuildInformation", return_value="  OpenCL: YES\n"):
            assert ModelManager._opencv_has_cuda_support() is False

    def test_exception_returns_false(self) -> None:
        with patch("cv2.getBuildInformation", side_effect=RuntimeError("fail")):
            assert ModelManager._opencv_has_cuda_support() is False
