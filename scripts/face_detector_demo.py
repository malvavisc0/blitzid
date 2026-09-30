"""FaceDetectorDNN demonstration script.

Showcases the public API exposed by :mod:`blitzid.detector`.

Run examples::

    # Default batch demo on ./images/
    python scripts/face_detector_demo.py

    # Single-image demos with the balanced preset
    python scripts/face_detector_demo.py --preset balanced --run all \
        --image images/bub_der_personalausweis_kopie.jpg
"""

from __future__ import annotations

import argparse
import logging
import time
from collections.abc import Iterable
from pathlib import Path

import cv2
from demo_cli import _parse_args

from blitzid import FaceDetectorDNN, ImageLoadError


def _build_detector(args: argparse.Namespace) -> FaceDetectorDNN:
    """Construct a detector from CLI arguments."""
    if args.preset == "fast":
        return FaceDetectorDNN.create_fast_detector(
            model_dir=args.model_dir,
            log_level=args.log_level,
        )

    if args.preset == "accurate":
        return FaceDetectorDNN.create_accurate_detector(
            model_dir=args.model_dir,
            log_level=args.log_level,
        )

    if args.preset == "balanced":
        return FaceDetectorDNN.create_balanced_detector(
            model_dir=args.model_dir,
            log_level=args.log_level,
        )

    if args.preset == "custom":
        return FaceDetectorDNN(
            confidence_threshold=args.confidence_threshold,
            min_face_size=tuple(args.min_face_size),
            nms_threshold=args.nms_threshold,
            log_level=args.log_level,
            enable_cache=args.cache,
            max_cache_size=args.max_cache_size,
            model_dir=args.model_dir,
        )

    raise ValueError(f"Unknown preset: {args.preset}")  # pragma: no cover


def _banner(title: str) -> None:
    print("\n" + "=" * 70)
    print(f"DEMO: {title}")
    print("=" * 70)


def demo_basic_detection(
    detector: FaceDetectorDNN,
    image_path: Path,
) -> None:
    """Single detection call — print per-face results."""
    _banner("Basic Detection")

    faces = detector.detect_face(image_path)
    print(f"Image: {image_path} | faces: {len(faces)}")
    for idx, (x, y, w, h, conf) in enumerate(faces, 1):
        print(f"  Face {idx}: x={x} y={y} w={w} h={h} confidence={conf:.2%}")


def demo_metrics(
    detector: FaceDetectorDNN,
    image_path: Path,
) -> None:
    """Show metrics output (miss → hit when caching is enabled)."""
    _banner("Detection With Metrics")

    detector.clear_cache()

    for label in ("First call", "Second call"):
        print(f"{label}:")
        r = detector.detect_face_with_metrics(image_path)
        print(f"  faces:              {r.num_faces}")
        print(f"  backend:            {r.backend}")
        print(f"  cache_hit:          {r.cache_hit}")
        print(f"  image_size:         {r.image_size}")
        print(f"  processing_time_ms: {r.processing_time * 1000:.2f}")


def demo_visualization(
    detector: FaceDetectorDNN,
    image_path: Path,
    results_dir: Path,
    label: str,
) -> None:
    """Draw detections and write visualisation images."""
    _banner("Visualization")

    results_dir.mkdir(parents=True, exist_ok=True)

    out_path = results_dir / f"{label}_visualization.jpg"
    result = detector.visualize_detections(
        image_path,
        output_path=out_path,
        show_confidence=True,
        color=(0, 255, 0),
        thickness=2,
    )
    print(f"Saved visualization: {out_path}")
    print(f"Output image shape:  {result.shape}")


def demo_face_extraction(
    detector: FaceDetectorDNN,
    image_path: Path,
    results_dir: Path,
    label: str,
    padding: float,
) -> None:
    """Extract face crops and save them."""
    _banner("Face Extraction")

    results_dir.mkdir(parents=True, exist_ok=True)
    extracted = detector.extract_faces(image_path, padding=padding)
    print(f"Extracted {len(extracted)} face crop(s)")

    for idx, (face_img, bbox, conf) in enumerate(extracted, 1):
        out = results_dir / f"{label}_face_{idx:03d}_conf_{conf:.2f}.jpg"
        ok = cv2.imwrite(str(out), face_img)
        print(
            f"  Face {idx}: bbox={bbox} conf={conf:.2%} "
            f"shape={face_img.shape} saved={ok} -> {out}"
        )


def demo_batch_processing(
    detector: FaceDetectorDNN,
    image_dir: Path,
    max_images: int,
) -> None:
    """Run detection over many images and print a summary."""
    _banner("Batch Processing")

    patterns = ("*.jpg", "*.jpeg", "*.png", "*.webp", "*.bmp", "*.tif", "*.tiff")
    paths: list[Path] = []
    for pat in patterns:
        paths.extend(p for p in image_dir.glob(pat) if p.is_file())

    paths = sorted({p.resolve() for p in paths})
    if max_images > 0:
        paths = paths[:max_images]

    image_path_args: list[str | Path] = list(paths)
    print(f"Batch input dir: {image_dir} | images: {len(paths)}")

    results = detector.detect_faces_batch(image_path_args, show_progress=True)

    total_faces = sum(len(v) for v in results.values())
    print("Batch summary:")
    print(f"  total_images: {len(results)}")
    print(f"  total_faces:  {total_faces}")


def demo_cache_speed(
    detector: FaceDetectorDNN,
    image_path: Path,
) -> None:
    """Crude timing demo: uncached → cached call."""
    _banner("Cache Speed")

    if not detector.cache_enabled:
        print("Cache disabled; skipping")
        return

    detector.clear_cache()

    start = time.time()
    detector.detect_face(image_path)
    t1 = time.time() - start

    start = time.time()
    detector.detect_face(image_path)
    t2 = time.time() - start

    print(f"First detection:  {t1 * 1000:.2f} ms")
    print(f"Second (cached):  {t2 * 1000:.2f} ms")
    if t2 > 0:
        print(f"Speedup:          {t1 / t2:.2f}x")


def demo_error_handling(
    detector: FaceDetectorDNN,
) -> None:
    """Demonstrate custom exceptions on invalid inputs."""
    _banner("Error Handling")

    try:
        detector.detect_face("nonexistent.jpg")
    except ImageLoadError as e:
        print(f"Caught ImageLoadError: {e}")


def _run_image_demos(
    detector: FaceDetectorDNN,
    run_list: list[str],
    image_path: Path,
    args: argparse.Namespace,
    label: str,
) -> None:
    """Run the demos that operate on a single image."""
    if "basic" in run_list:
        demo_basic_detection(detector, image_path)
    if "metrics" in run_list:
        demo_metrics(detector, image_path)
    if "visualize" in run_list:
        demo_visualization(detector, image_path, args.results_dir, label)
    if "extract" in run_list:
        demo_face_extraction(
            detector, image_path, args.results_dir, label, padding=args.padding
        )
    if "cache" in run_list:
        demo_cache_speed(detector, image_path)


def _run_demos(
    detector: FaceDetectorDNN,
    run_list: list[str],
    image_path: Path | None,
    args: argparse.Namespace,
    label: str,
) -> None:
    """Run the selected demos."""
    if image_path is not None:
        _run_image_demos(detector, run_list, image_path, args, label)
    if "batch" in run_list:
        demo_batch_processing(detector, args.batch_dir, args.max_images)
    if "errors" in run_list:
        demo_error_handling(detector)


def main(argv: Iterable[str] | None = None) -> None:
    """CLI entrypoint."""
    args, run_list, image_path = _parse_args(list(argv) if argv is not None else None)

    logging.basicConfig(level=args.log_level, format="%(levelname)s - %(message)s")

    detector = _build_detector(args)

    cache_on = detector.cache_enabled
    label = f"{args.preset}_cache_{'on' if cache_on else 'off'}"

    _run_demos(detector, run_list, image_path, args, label)


if __name__ == "__main__":
    main()
