"""RapidOCRReader demonstration script.

Run example::

    python scripts/ocr_demo.py --image images/bub_der_personalausweis_kopie.jpg

Recognized lines are printed and written to the results directory
(default: ``results/``, a local artifact — not committed).
"""

from __future__ import annotations

import argparse
import logging
import time
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
        "--results-dir",
        type=Path,
        default=Path("results"),
        help="Directory to write the recognized lines into.",
    )
    parser.add_argument(
        "--log-level",
        type=int,
        default=logging.WARNING,
        help="Logging level (default: WARNING).",
    )
    return parser.parse_args(argv)


def _format_lines(
    image: Path, texts: list[tuple[tuple[int, int, int, int], str, float]]
) -> list[str]:
    """Render the recognized lines as markdown."""
    lines = [f"# OCR — {image.name}", "", f"text lines: {len(texts)}", ""]
    lines.extend(
        f"{idx}: ({x}, {y}, {w}x{h}) conf={conf:.2%} {text}"
        for idx, ((x, y, w, h), text, conf) in enumerate(texts, 1)
    )
    return lines


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

    args.results_dir.mkdir(parents=True, exist_ok=True)
    stamp = time.strftime("%Y%m%d-%H%M%S")
    out_path = args.results_dir / f"ocr-{args.image.stem}-{stamp}.md"
    records = [(line.bbox, line.text, line.confidence) for line in texts]
    out_path.write_text("\n".join(_format_lines(args.image, records)) + "\n")
    print(f"Results written to {out_path}")


if __name__ == "__main__":
    main()
