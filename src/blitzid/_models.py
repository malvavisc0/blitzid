"""Model management for face detection DNN.

Combines model download/load logic with default path resolution.
Always uses ``platformdirs`` for model storage; the dev-install
heuristic has been removed for reliability.
"""

from __future__ import annotations

import logging
import urllib.error
import urllib.request
from pathlib import Path

import cv2

from .exceptions import ModelError

# ---------------------------------------------------------------------------
# Default model directory
# ---------------------------------------------------------------------------


def default_model_dir(subdir: str | None = None) -> Path:
    """Return the default directory for storing model weights.

    Uses ``platformdirs.user_cache_dir("blitzid")`` in all environments.

    Args:
        subdir: Optional subdirectory (e.g. ``"deepface"``).

    Returns:
        Resolved :class:`~pathlib.Path` that is guaranteed to exist.
    """
    try:
        from platformdirs import user_cache_dir
    except ImportError:  # pragma: no cover
        base = Path.home() / ".cache" / "blitzid" / "models"
    else:
        base = Path(user_cache_dir("blitzid")) / "models"

    if subdir:
        base = base / subdir

    base.mkdir(parents=True, exist_ok=True)
    return base


# ---------------------------------------------------------------------------
# Model manager
# ---------------------------------------------------------------------------


class ModelManager:
    """Manages DNN model files and network loading."""

    PROTO_URL = (
        "https://raw.githubusercontent.com/opencv/opencv/master/"
        "samples/dnn/face_detector/deploy.prototxt"
    )
    MODEL_URL = (
        "https://github.com/opencv/opencv_3rdparty/raw/"
        "dnn_samples_face_detector_20170830/"
        "res10_300x300_ssd_iter_140000.caffemodel"
    )

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

        self.proto_path = self.model_dir / "deploy.prototxt"
        self.model_path = self.model_dir / "res10_300x300_ssd_iter_140000.caffemodel"

    def ensure_models_exist(self) -> None:
        """Ensure model files exist, downloading if allowed."""
        missing = [p for p in (self.proto_path, self.model_path) if not p.exists()]

        if not missing:
            return

        if not self.allow_downloads:
            names = ", ".join(p.name for p in missing)
            raise ModelError(
                f"Missing model files: {names}. "
                "Downloads are disabled; set allow_downloads=True."
            )

        self.logger.info("DNN models not found. Downloading...")
        self._download_models()

    @staticmethod
    def _opencv_has_cuda_support() -> bool:
        """Best-effort check for whether OpenCV was built with CUDA."""
        try:
            info = cv2.getBuildInformation()
        except Exception:
            return False

        for line in info.splitlines():
            s = line.strip()
            if s.startswith("NVIDIA CUDA") or s.startswith("CUDA"):
                return "YES" in s.upper()

        return False

    def load_network(
        self, use_cuda: bool = True, require_cuda: bool = False
    ) -> tuple[cv2.dnn.Net, str]:
        """Load network and configure backend.

        Returns:
            Tuple of (network, backend_type) where backend_type is "CUDA" or "CPU"

        Raises:
            ModelError: If models cannot be loaded or CUDA is required but unavailable.
        """
        self.ensure_models_exist()

        self.logger.info("Loading DNN network for face detection...")
        net = cv2.dnn.readNetFromCaffe(str(self.proto_path), str(self.model_path))

        backend_type = "CPU"

        if use_cuda:
            if not self._opencv_has_cuda_support():
                msg = "OpenCV was built without CUDA support"
                if require_cuda:
                    raise ModelError(msg)
                self.logger.info("%s; using CPU", msg)
                net.setPreferableBackend(cv2.dnn.DNN_BACKEND_DEFAULT)
                net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
            else:
                try:
                    cuda_count = cv2.cuda.getCudaEnabledDeviceCount()
                except Exception as e:  # pragma: no cover
                    msg = f"OpenCV CUDA module unavailable: {e}"
                    if require_cuda:
                        raise ModelError(msg) from e
                    self.logger.warning(
                        "OpenCV CUDA module unavailable (falling back to CPU): %s",
                        e,
                    )
                    cuda_count = 0

                self.logger.info("Available CUDA devices: %d", cuda_count)

                if cuda_count > 0:
                    try:
                        net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
                        net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)
                        backend_type = "CUDA"
                        self.logger.info("DNN configured to use CUDA backend")
                    except cv2.error as e:
                        msg = f"Error configuring CUDA backend/target: {e}"
                        if require_cuda:
                            raise ModelError(msg) from e
                        self.logger.warning(
                            "Error configuring CUDA (falling back to CPU): %s",
                            e,
                        )
                        net.setPreferableBackend(cv2.dnn.DNN_BACKEND_DEFAULT)
                        net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
                else:
                    msg = (
                        "OpenCV has CUDA support but no CUDA device is visible "
                        "(getCudaEnabledDeviceCount() == 0)"
                    )
                    if require_cuda:
                        raise ModelError(msg)
                    self.logger.info("%s; using CPU", msg)
                    net.setPreferableBackend(cv2.dnn.DNN_BACKEND_DEFAULT)
                    net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
        else:
            net.setPreferableBackend(cv2.dnn.DNN_BACKEND_DEFAULT)
            net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
            self.logger.info("Using CPU backend")

        return net, backend_type

    def _download_models(self) -> None:
        """Download model files if they don't exist.

        Raises:
            ModelError: If download fails
        """
        urls = {
            self.proto_path: self.PROTO_URL,
            self.model_path: self.MODEL_URL,
        }
        headers = {"User-Agent": "Mozilla/5.0 (compatible; OpenCV DNN Face Detector)"}

        for path, url in urls.items():
            if path.exists():
                self.logger.info("%s already exists", path.name)
                continue

            self.logger.info("Downloading %s...", path.name)
            tmp_path = path.with_suffix(path.suffix + ".tmp")
            try:
                req = urllib.request.Request(url, headers=headers)
                with urllib.request.urlopen(
                    req, timeout=self.download_timeout
                ) as response:
                    data = response.read()

                # Write atomically-ish: write then replace.
                with open(tmp_path, "wb") as out_file:
                    out_file.write(data)
                tmp_path.replace(path)

                self.logger.info(
                    "%s downloaded (%.1f KB)",
                    path.name,
                    len(data) / 1024,
                )
            except (
                urllib.error.URLError,
                TimeoutError,
                OSError,
                ValueError,
            ) as e:
                # Clean up partial files on failure.
                tmp_path.unlink(missing_ok=True)
                path.unlink(missing_ok=True)
                raise ModelError(
                    f"Error downloading {path.name} from {url} "
                    f"(timeout={self.download_timeout}s): {e}"
                ) from e
