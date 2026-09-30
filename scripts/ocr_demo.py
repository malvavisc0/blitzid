"""RapidOCRReader demonstration script.

Run example::

    python scripts/ocr_demo.py --image images/bub_der_personalausweis_kopie.jpg
"""

from __future__ import annotations

import argparse
import logging
from collections.abc import Iterable
from pathlib import Path

from blitzid import RapidOCRReader


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Read text lines from an image.")
    parser.add_argument(
        "--image",
        type=Path,
        default=Path("images/bub_der_personalausweis_kopie.jpg"),
        help="Image to read (path).",
    )
    parser.add_argument(
        "--log-level",
        type=int,
        default=logging.WARNING,
        help="Logging level (default: WARNING).",
    )
    return parser.parse_args(argv)


def main(argv: Iterable[str] | None = None) -> None:
    """CLI entrypoint."""
    args = _parse_args(list(argv) if argv is not None else None)
    logging.basicConfig(level=args.log_level, format="%(levelname)s - %(message)s")

    reader = RapidOCRReader(log_level=args.log_level)
    texts = reader.read(args.image)

    print(f"Image: {args.image} | text lines: {len(texts)}")
    for idx, line in enumerate(texts, 1):
        x, y, w, h = line.bbox
        print(f"  {idx}: ({x}, {y}, {w}x{h}) conf={line.confidence:.2%} {line.text}")


if __name__ == "__main__":
    main()
