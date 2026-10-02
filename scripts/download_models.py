"""Download all model weights into a local models directory.

Run examples::

    python scripts/download_models.py
    python scripts/download_models.py --models-dir /models

SCRFD (face detection) and ArcFace (face recognition) weights always
download. RapidOCR weights (the ``ocr`` extra) download when the extra
is installed; otherwise they are skipped with a note.
"""

from __future__ import annotations

import argparse
import logging
from collections.abc import Iterable
from pathlib import Path

from blitzid import RapidOCRReader
from blitzid._models import ModelManager
from blitzid.face._arcface import (
    ARCFACE_MODEL_FILENAME,
    ARCFACE_MODEL_SHA256,
    ARCFACE_MODEL_URL,
)
from blitzid.face._attributes import (
    ATTRIBUTE_MODEL_FILENAME,
    ATTRIBUTE_MODEL_SHA256,
    ATTRIBUTE_MODEL_URL,
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


def _download_scrfd(models_dir: Path) -> None:
    """Download the SCRFD detector weights if missing."""
    manager = ModelManager(models_dir, logging.getLogger("download_models"))
    manager.ensure_model_exists()
    size_kb = manager.model_path.stat().st_size / 1024
    print(f"SCRFD detector: {manager.model_path} ({size_kb:.0f} KB)")


def _download_arcface(models_dir: Path) -> None:
    """Download the ArcFace recognition weights if missing."""
    manager = ModelManager(
        models_dir,
        logging.getLogger("download_models"),
        filename=ARCFACE_MODEL_FILENAME,
        url=ARCFACE_MODEL_URL,
        sha256=ARCFACE_MODEL_SHA256,
    )
    manager.ensure_model_exists()
    size_kb = manager.model_path.stat().st_size / 1024
    print(f"ArcFace recognizer: {manager.model_path} ({size_kb:.0f} KB)")


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


def _download_rapidocr(models_dir: Path) -> None:
    """Download the RapidOCR text-reading weights if the extra is installed."""
    try:
        import rapidocr  # noqa: F401
    except ImportError:
        print("(rapidocr models skipped: blitzid[ocr] extra not installed)")
        return

    RapidOCRReader(model_dir=models_dir / "rapidocr")
    print(f"RapidOCR models: {models_dir / 'rapidocr'}")


def main(argv: Iterable[str] | None = None) -> None:
    """CLI entrypoint."""
    args = _parse_args(list(argv) if argv is not None else None)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s - %(message)s")
    _download_scrfd(args.models_dir)
    _download_arcface(args.models_dir)
    _download_attributes(args.models_dir)
    _download_rapidocr(args.models_dir)


if __name__ == "__main__":
    main()
