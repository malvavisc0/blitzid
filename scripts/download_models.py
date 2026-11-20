"""Download all model weights into a local models directory.

Run examples::

    python scripts/download_models.py
    python scripts/download_models.py --models-dir /models

YuNet (face detection), SFace (face recognition), and RapidOCR text
weights always download (all are core dependencies).
"""

from __future__ import annotations

import argparse
import logging
import tempfile
from collections.abc import Iterable
from pathlib import Path

from blitzid._models import ModelManager
from blitzid.face._antispoof import (
    ANTISPOOF_MODEL_FILENAME,
    ANTISPOOF_MODEL_SHA256,
    ANTISPOOF_MODEL_URL,
)
from blitzid.face._attributes import (
    ATTRIBUTE_MODEL_FILENAME,
    ATTRIBUTE_MODEL_SHA256,
    ATTRIBUTE_MODEL_URL,
)
from blitzid.face._sface import (
    SFACE_MODEL_FILENAME,
    SFACE_MODEL_SHA256,
    SFACE_MODEL_URL,
)
from blitzid.face._yunet import (
    YUNET_MODEL_FILENAME,
    YUNET_MODEL_SHA256,
    YUNET_MODEL_URL,
)


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download all blitzid model weights.")
    parser.add_argument(
        "--models-dir",
        type=Path,
        default=Path("models"),
        help='Directory to download weights into (default: "models").',
    )
    return parser.parse_args(argv)


def _download_yunet(models_dir: Path) -> None:
    """Download the YuNet detector weights if missing."""
    manager = ModelManager(
        models_dir,
        logging.getLogger("download_models"),
        filename=YUNET_MODEL_FILENAME,
        url=YUNET_MODEL_URL,
        sha256=YUNET_MODEL_SHA256,
    )
    manager.ensure_model_exists()
    size_kb = manager.model_path.stat().st_size / 1024
    print(f"YuNet detector: {manager.model_path} ({size_kb:.0f} KB)")


def _download_sface(models_dir: Path) -> None:
    """Download the SFace recognition weights if missing."""
    manager = ModelManager(
        models_dir,
        logging.getLogger("download_models"),
        filename=SFACE_MODEL_FILENAME,
        url=SFACE_MODEL_URL,
        sha256=SFACE_MODEL_SHA256,
    )
    manager.ensure_model_exists()
    size_kb = manager.model_path.stat().st_size / 1024
    print(f"SFace recognizer: {manager.model_path} ({size_kb:.0f} KB)")


def _download_attributes(models_dir: Path) -> None:
    """Download the FairFace attribute weights if missing."""
    manager = ModelManager(
        models_dir,
        logging.getLogger("download_models"),
        filename=ATTRIBUTE_MODEL_FILENAME,
        url=ATTRIBUTE_MODEL_URL,
        sha256=ATTRIBUTE_MODEL_SHA256,
    )
    manager.ensure_model_exists()
    size_kb = manager.model_path.stat().st_size / 1024
    print(f"FairFace attributes: {manager.model_path} ({size_kb:.0f} KB)")


def _download_antispoof(models_dir: Path) -> None:
    """Download the MiniFASNet anti-spoof weights if missing."""
    manager = ModelManager(
        models_dir,
        logging.getLogger("download_models"),
        filename=ANTISPOOF_MODEL_FILENAME,
        url=ANTISPOOF_MODEL_URL,
        sha256=ANTISPOOF_MODEL_SHA256,
    )
    manager.ensure_model_exists()
    size_kb = manager.model_path.stat().st_size / 1024
    print(f"MiniFASNet anti-spoof: {manager.model_path} ({size_kb:.0f} KB)")


def _download_rapidocr(models_dir: Path) -> None:
    """Download every RapidOCR text-reading weight."""
    from omegaconf import OmegaConf
    from rapidocr.main import DEFAULT_CFG_PATH
    from rapidocr.utils.download_models import download_models

    target = models_dir / "rapidocr"
    cfg = OmegaConf.load(DEFAULT_CFG_PATH)
    cfg.Global.model_root_dir = str(target)
    with tempfile.TemporaryDirectory() as temp_dir:
        config_path = Path(temp_dir) / "rapidocr.yaml"
        OmegaConf.save(cfg, config_path)
        download_models(config_path)
    print(f"RapidOCR models: {target}")


def main(argv: Iterable[str] | None = None) -> None:
    """CLI entrypoint."""
    args = _parse_args(list(argv) if argv is not None else None)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s - %(message)s")
    _download_yunet(args.models_dir)
    _download_sface(args.models_dir)
    _download_attributes(args.models_dir)
    _download_antispoof(args.models_dir)
    _download_rapidocr(args.models_dir)


if __name__ == "__main__":
    main()
