"""Model management for ONNX model weights.

Downloads model files (defaulting to the SCRFD-2.5G detector, InsightFace
``buffalo_m`` detection weights) and creates a CPU inference session.
Custom files (e.g. the ArcFace recognition model) are selected via the
``filename`` / ``url`` / ``sha256`` constructor arguments; downloads are
verified against the pinned digest (pass ``sha256=""`` to skip the
check for a custom model). Weights live in the default models dir:
``BLITZID_MODELS_DIR`` when set, else the ``platformdirs`` user cache.
"""

from __future__ import annotations

import hashlib
import logging
import os
import urllib.request
from pathlib import Path

import onnxruntime as ort  # type: ignore[import-untyped]
from platformdirs import user_cache_dir

from blitzid.exceptions import ModelError

_DOWNLOAD_CHUNK_BYTES = 64 * 1024

MODELS_DIR_ENV = "BLITZID_MODELS_DIR"


def session_options() -> ort.SessionOptions:
    """Session options shared by every in-house ONNX engine.

    Intra-op threads do not spin-wait between kernels: the engines run
    back to back (detector, then recognizer or classifier) and several
    sessions share one process in the API, so a pool busy-spinning after
    its last op only steals cores from the next session. Measured on a
    16-thread CPU: detect + embed drops from ~260 ms to ~180 ms, with no
    change to any output.
    """
    options = ort.SessionOptions()
    options.add_session_config_entry("session.intra_op.allow_spinning", "0")
    return options


def default_model_dir(subdir: str | None = None) -> Path:
    """Return the default directory for storing model weights.

    Honors ``BLITZID_MODELS_DIR``; otherwise uses
    ``platformdirs.user_cache_dir("blitzid")``.

    Args:
        subdir: Optional subdirectory.

    Returns:
        Resolved :class:`~pathlib.Path` that is guaranteed to exist.
    """
    env_dir = os.environ.get(MODELS_DIR_ENV)
    base = Path(env_dir) if env_dir else Path(user_cache_dir("blitzid")) / "models"

    if subdir:
        base = base / subdir

    base.mkdir(parents=True, exist_ok=True)
    return base


class ModelManager:
    """Manages an ONNX model file and its inference session."""

    MODEL_URL = (
        "https://huggingface.co/immich-app/buffalo_m/resolve/main/detection/model.onnx"
    )
    MODEL_FILENAME = "scrfd_2.5g.onnx"
    MODEL_SHA256 = "041f73f47371333d1d17a6fee6c8ab4e6aecabefe398ff32cca4e2d5eaee0af9"

    def __init__(
        self,
        model_dir: Path,
        logger: logging.Logger,
        allow_downloads: bool = True,
        download_timeout: float = 30.0,
        filename: str | None = None,
        url: str | None = None,
        sha256: str | None = None,
    ):
        self.model_dir = Path(model_dir)
        self.model_dir.mkdir(exist_ok=True, parents=True)
        self.logger = logger
        self.allow_downloads = allow_downloads
        self.download_timeout = download_timeout

        self.filename = filename if filename is not None else self.MODEL_FILENAME
        self.url = url if url is not None else self.MODEL_URL
        self.sha256 = sha256 if sha256 is not None else self.MODEL_SHA256
        self.model_path = self.model_dir / self.filename

    def ensure_model_exists(self) -> None:
        """Ensure the model file exists, downloading if allowed.

        Raises:
            ModelError: If the model is missing and downloads are disabled.
        """
        if self.model_path.exists():
            return

        if not self.allow_downloads:
            raise ModelError(
                f"Missing model file: {self.model_path}. "
                "Downloads are disabled; set allow_downloads=True."
            )

        self.logger.info("%s not found. Downloading...", self.filename)
        self._download_model()

    def load_session(self) -> ort.InferenceSession:
        """Load the model as a CPU inference session.

        Raises:
            ModelError: If the model cannot be loaded.
        """
        self.ensure_model_exists()

        self.logger.info("Loading %s...", self.filename)
        try:
            return ort.InferenceSession(
                str(self.model_path),
                sess_options=session_options(),
                providers=["CPUExecutionProvider"],
            )
        except Exception as e:
            raise ModelError(f"Failed to load model {self.model_path.name}: {e}") from e

    def _download_model(self) -> None:
        """Download the model file.

        Streams to a ``.tmp`` file, verifies it against the pinned
        digest, and renames into place on success, so a partial or
        corrupted download never masquerades as a valid model.

        Raises:
            ModelError: If the download or digest verification fails.
        """
        path = self.model_path
        url = self.url
        headers = {"User-Agent": "Mozilla/5.0 (compatible; blitzid)"}

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

            self._verify_digest(tmp_path, url)

            tmp_path.replace(path)
            self.logger.info(
                "%s downloaded (%.1f KB)", path.name, path.stat().st_size / 1024
            )
        except (OSError, ValueError) as e:
            tmp_path.unlink(missing_ok=True)
            raise ModelError(
                f"Error downloading {path.name} from {url} "
                f"(timeout={self.download_timeout}s): {e}"
            ) from e
        except ModelError:
            tmp_path.unlink(missing_ok=True)
            raise

    def _verify_digest(self, tmp_path: Path, url: str) -> None:
        """Fail when a downloaded file does not match the pinned digest.

        An empty ``sha256`` skips the check (custom models without a
        known digest).
        """
        if not self.sha256:
            return

        digest = hashlib.sha256()
        with open(tmp_path, "rb") as f:
            while chunk := f.read(_DOWNLOAD_CHUNK_BYTES):
                digest.update(chunk)
        actual = digest.hexdigest()
        if actual != self.sha256:
            raise ModelError(
                f"Digest mismatch for {tmp_path.name}: expected "
                f"{self.sha256}, got {actual} (from {url})"
            )
