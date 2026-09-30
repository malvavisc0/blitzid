"""CLI plumbing for face_detector_demo — argument parsing and dispatch."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

_KNOWN_DEMOS = (
    "basic",
    "metrics",
    "visualize",
    "extract",
    "batch",
    "cache",
    "errors",
)

_IMAGE_DEMOS = ("basic", "metrics", "visualize", "extract", "cache")


def _parse_log_level(value: str) -> int:
    """Parse a log-level string (e.g. ``"INFO"``) into a :mod:`logging` constant."""
    mapping = {
        "CRITICAL": logging.CRITICAL,
        "ERROR": logging.ERROR,
        "WARNING": logging.WARNING,
        "INFO": logging.INFO,
        "DEBUG": logging.DEBUG,
    }
    try:
        return mapping[value.upper()]
    except KeyError as e:
        raise argparse.ArgumentTypeError(
            f"Invalid log level: {value}. Choose from: {', '.join(mapping)}"
        ) from e


def _parse_run_list(value: str) -> list[str]:
    """Parse ``--run`` into a validated list of demo names."""
    value = value.strip().lower()
    if value == "all":
        return list(_KNOWN_DEMOS)
    items = [v.strip().lower() for v in value.split(",") if v.strip()]
    unknown = [v for v in items if v not in _KNOWN_DEMOS]
    if unknown:
        raise SystemExit(
            f"Unknown demo(s): {', '.join(unknown)}. "
            f"Choose from: {', '.join(_KNOWN_DEMOS)} (or 'all')"
        )
    return items


def _build_parser() -> argparse.ArgumentParser:
    """Build the CLI argument parser."""
    parser = argparse.ArgumentParser(description="FaceDetectorDNN demo")

    parser.add_argument(
        "--preset",
        choices=["balanced", "fast", "accurate", "custom"],
        default="balanced",
        help="Detector preset",
    )
    parser.add_argument(
        "--run",
        default="batch",
        help=(
            "Comma-separated demos: "
            "basic,metrics,visualize,extract,batch,cache,errors "
            "or 'all'"
        ),
    )
    parser.add_argument(
        "--image",
        type=Path,
        default=None,
        help="Path to a test image (required for single-image demos)",
    )
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=Path("results"),
        help="Directory for output images",
    )
    parser.add_argument(
        "--batch-dir",
        type=Path,
        default=Path("images"),
        help="Directory for batch processing",
    )
    parser.add_argument(
        "--max-images",
        type=int,
        default=0,
        help="Limit batch image count (0 = no limit)",
    )
    parser.add_argument(
        "--padding",
        type=float,
        default=0.2,
        help="Face crop padding fraction (0.0-1.0)",
    )
    parser.add_argument(
        "--cache",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="(custom preset) enable/disable caching",
    )
    parser.add_argument(
        "--max-cache-size",
        type=int,
        default=200,
        help="(custom preset) maximum cache entries",
    )
    parser.add_argument(
        "--log-level",
        type=_parse_log_level,
        default=logging.INFO,
        help="Logging level (DEBUG, INFO, WARNING, ERROR)",
    )
    parser.add_argument(
        "--model-dir",
        type=Path,
        default=None,
        help="Model directory (default uses platformdirs cache)",
    )

    parser.add_argument(
        "--confidence-threshold",
        type=float,
        default=0.5,
        help="(custom preset) confidence threshold (0.0-1.0)",
    )
    parser.add_argument(
        "--min-face-size",
        type=int,
        nargs=2,
        default=(50, 50),
        metavar=("W", "H"),
        help="(custom preset) minimum face size in pixels",
    )
    parser.add_argument(
        "--nms-threshold",
        type=float,
        default=0.3,
        help="(custom preset) NMS IoU threshold (0.0-1.0)",
    )

    return parser


def _resolve_image(args: argparse.Namespace, run_list: list[str]) -> Path | None:
    """Return the input image path, validating it when demos need one."""
    image: Path | None = args.image
    needs_image = any(n in run_list for n in _IMAGE_DEMOS)
    if not needs_image:
        return image
    if image is None:
        raise SystemExit(
            "--image is required for the selected demos "
            "(basic/metrics/visualize/extract/cache)"
        )
    if not image.exists():
        raise SystemExit(f"Image not found: {image}")
    return image


def _parse_args(
    argv: list[str] | None,
) -> tuple[argparse.Namespace, list[str], Path | None]:
    """Parse CLI arguments and return (args, run_list, image_path)."""
    args = _build_parser().parse_args(argv)
    run_list = _parse_run_list(args.run)
    image_path = _resolve_image(args, run_list)
    return args, run_list, image_path
