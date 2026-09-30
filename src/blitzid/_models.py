"""Model management for the SCRFD face detection network.

Downloads the SCRFD-2.5G ONNX model (InsightFace ``buffalo_m`` detection
weights) and creates a CPU inference session. Always uses ``platformdirs``
for model storage.
"""

from __future__ import annotations

import logging
import urllib.error
import urllib.request
from pathlib import Path

import onnxruntime as ort  # type: ignore[import-untyped]
from platformdirs import user_cache_dir

from .exceptions import ModelError

_DOWNLOAD_CHUNK_BYTES = 64 * 1024


def default_model_dir(subdir: str | None = None) -> Path:
    """Return the default directory for storing model weights.

    Uses ``platformdirs.user_cache_dir("blitzid")``.

    Args:
        subdir: Optional subdirectory.

    Returns:
        Resolved :class:`~pathlib.Path` that is guaranteed to exist.
    """
    base = Path(user_cache_dir("blitzid")) / "models"

    if subdir:
        base = base / subdir

    base.mkdir(parents=True, exist_ok=True)
    return base


class ModelManager:
    """Manages the SCRFD ONNX model file and inference session."""

    MODEL_URL = (
        "https://huggingface.co/immich-app/buffalo_m/resolve/main/detection/model.onnx"
    )
    MODEL_FILENAME = "scrfd_2.5g.onnx"

    def __init__(
        self,
        model_dir: Path,
        logger: logging.Logger,
        allow_downloads: bool = True,
        download_timeout: float = 30.0,
    ):
        self.model_dir = Path(model_dir)
        self.model_dir.mkdir(exist_ok=True, parents=True)
        self.logger = logger
        self.allow_downloads = allow_downloads
        self.download_timeout = download_timeout

        self.model_path = self.model_dir / self.MODEL_FILENAME

    def ensure_model_exists(self) -> None:
        """Ensure the model file exists, downloading if allowed.

        Raises:
            ModelError: If the model is missing and downloads are disabled.
        """
        if self.model_path.exists():
            return

        if not self.allow_downloads:
            raise ModelError(
                f"Missing model file: {self.model_path.name}. "
                "Downloads are disabled; set allow_downloads=True."
            )

        self.logger.info("SCRFD model not found. Downloading...")
        self._download_model()

    def load_session(self) -> ort.InferenceSession:
        """Load the SCRFD model as a CPU inference session.

        Raises:
            ModelError: If the model cannot be loaded.
        """
        self.ensure_model_exists()

        self.logger.info("Loading SCRFD face detection model...")
        try:
            return ort.InferenceSession(
                str(self.model_path), providers=["CPUExecutionProvider"]
            )
        except Exception as e:
            raise ModelError(f"Failed to load model {self.model_path.name}: {e}") from e

    def _download_model(self) -> None:
        """Download the model file.

        Streams to a ``.tmp`` file and renames into place on success, so
        a partial download never masquerades as a valid model.

        Raises:
            ModelError: If the download fails.
        """
        path = self.model_path
        url = self.MODEL_URL
        headers = {"User-Agent": "Mozilla/5.0 (compatible; blitzid face detector)"}

        self.logger.info("Downloading %s...", path.name)
        tmp_path = path.with_suffix(path.suffix + ".tmp")
        try:
            req = urllib.request.Request(url, headers=headers)
            with (
                urllib.request.urlopen(req, timeout=self.download_timeout) as response,
                open(tmp_path, "wb") as out_file,
            ):
                while True:
                    chunk = response.read(_DOWNLOAD_CHUNK_BYTES)
                    if not chunk:
                        break
                    out_file.write(chunk)

            tmp_path.replace(path)
            self.logger.info(
                "%s downloaded (%.1f KB)", path.name, path.stat().st_size / 1024
            )
        except (urllib.error.URLError, TimeoutError, OSError, ValueError) as e:
            tmp_path.unlink(missing_ok=True)
            raise ModelError(
                f"Error downloading {path.name} from {url} "
                f"(timeout={self.download_timeout}s): {e}"
            ) from e
