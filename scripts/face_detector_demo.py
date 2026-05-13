"""FaceDetectorDNN demonstration script.

Showcases the public API exposed by :mod:`blitzid.detector` and the
optional :class:`~blitzid.deepface.FaceDetectorDeepFace` backend.

Run examples::

    # Default batch demo on ./images/
    python scripts/face_detector_demo.py

    # Single-image demos with the balanced preset
    python scripts/face_detector_demo.py --preset balanced --run all \
        --image images/bub_der_personalausweis_kopie.jpg

    # DeepFace backend
    python scripts/face_detector_demo.py --preset deepface --run basic,extract \
        --image images/bub_der_personalausweis_kopie.jpg
"""

from __future__ import annotations

import argparse
import logging
import time
from collections.abc import Iterable
from pathlib import Path

import cv2

from blitzid import FaceDetectorDeepFace, FaceDetectorDNN, ImageLoadError

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


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
    """Parse ``--run`` into a list of demo names."""
    value = value.strip().lower()
    if value == "all":
        return [
            "basic",
            "metrics",
            "visualize",
            "extract",
            "analyze",
            "batch",
            "cache",
            "errors",
        ]
    return [v.strip().lower() for v in value.split(",") if v.strip()]


# ---------------------------------------------------------------------------
# Detector construction
# ---------------------------------------------------------------------------


def _build_detector(args: argparse.Namespace) -> FaceDetectorDNN | FaceDetectorDeepFace:
    """Construct a detector from CLI arguments."""

    if args.preset == "fast":
        return FaceDetectorDNN.create_fast_detector(
            model_dir=args.model_dir,
            log_level=args.log_level,
            require_cuda=args.require_cuda,
        )

    if args.preset == "accurate":
        return FaceDetectorDNN.create_accurate_detector(
            model_dir=args.model_dir,
            log_level=args.log_level,
            require_cuda=args.require_cuda,
        )

    if args.preset == "balanced":
        return FaceDetectorDNN.create_balanced_detector(
            model_dir=args.model_dir,
            log_level=args.log_level,
            require_cuda=args.require_cuda,
        )

    if args.preset == "deepface":
        detector = FaceDetectorDeepFace(
            model_dir=args.model_dir,
            log_level=args.log_level,
            detector_backend="retinaface",
            align=True,
            enforce_detection=False,
        )
        if args.require_cuda:
            try:
                import tensorflow as tf  # type: ignore[import-untyped]

                gpus = tf.config.list_physical_devices("GPU")
            except Exception as exc:
                raise SystemExit(
                    f"--require-cuda set but TensorFlow GPU unavailable: {exc}"
                ) from exc
            if not gpus:
                raise SystemExit(
                    "--require-cuda set but TensorFlow reports no GPU devices"
                )
        return detector

    if args.preset == "custom":
        return FaceDetectorDNN(
            confidence_threshold=args.confidence_threshold,
            min_face_size=tuple(args.min_face_size),
            nms_threshold=args.nms_threshold,
            log_level=args.log_level,
            enable_cache=args.cache,
            max_cache_size=args.max_cache_size,
            model_dir=args.model_dir,
            require_cuda=args.require_cuda,
        )

    raise ValueError(f"Unknown preset: {args.preset}")  # pragma: no cover


# ---------------------------------------------------------------------------
# Demo functions
# ---------------------------------------------------------------------------


def demo_basic_detection(
    detector: FaceDetectorDNN | FaceDetectorDeepFace,
    image_path: Path,
) -> None:
    """Single detection call — print per-face results."""
    print("\n" + "=" * 70)
    print("DEMO: Basic Detection")
    print("=" * 70)

    faces = detector.detect_face(image_path)
    print(f"Image: {image_path} | faces: {len(faces)}")
    for idx, (x, y, w, h, conf) in enumerate(faces, 1):
        print(f"  Face {idx}: x={x} y={y} w={w} h={h} confidence={conf:.2%}")


def demo_metrics(
    detector: FaceDetectorDNN | FaceDetectorDeepFace,
    image_path: Path,
) -> None:
    """Show metrics output (miss → hit when caching is enabled)."""
    print("\n" + "=" * 70)
    print("DEMO: Detection With Metrics")
    print("=" * 70)

    # Clear cache so the first call is a guaranteed miss.
    if isinstance(detector, FaceDetectorDNN):
        detector.clear_cache()

    r1 = detector.detect_face_with_metrics(image_path)
    r2 = detector.detect_face_with_metrics(image_path)

    for label, r in [("First call", r1), ("Second call", r2)]:
        print(f"{label}:")
        print(f"  faces:              {r.num_faces}")
        print(f"  backend:            {r.backend}")
        print(f"  cache_hit:          {r.cache_hit}")
        print(f"  image_size:         {r.image_size}")
        print(f"  processing_time_ms: {r.processing_time * 1000:.2f}")


def demo_visualization(
    detector: FaceDetectorDNN | FaceDetectorDeepFace,
    image_path: Path,
    results_dir: Path,
    label: str,
) -> None:
    """Draw detections and write visualisation images."""
    print("\n" + "=" * 70)
    print("DEMO: Visualization")
    print("=" * 70)

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
    detector: FaceDetectorDNN | FaceDetectorDeepFace,
    image_path: Path,
    results_dir: Path,
    label: str,
    padding: float,
) -> None:
    """Extract face crops and save them."""
    print("\n" + "=" * 70)
    print("DEMO: Face Extraction")
    print("=" * 70)

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


def demo_deepface_analyze_on_crop(
    detector: FaceDetectorDNN | FaceDetectorDeepFace,
    image_path: Path,
    results_dir: Path,
    label: str,
) -> None:
    """Save a cropped face and run ``analyze()`` (DeepFace only)."""
    print("\n" + "=" * 70)
    print("DEMO: DeepFace Analyze on Saved Crop")
    print("=" * 70)

    if not isinstance(detector, FaceDetectorDeepFace):
        print("analyze() not supported by this backend; skipping")
        return

    results_dir.mkdir(parents=True, exist_ok=True)
    extracted = detector.extract_faces(image_path, padding=0.0)
    if not extracted:
        print("No faces extracted; skipping")
        return

    face_img, _bbox, conf = extracted[0]
    crop_path = results_dir / f"{label}_crop_for_analyze_conf_{conf:.2f}.jpg"
    cv2.imwrite(str(crop_path), face_img)
    print(f"Wrote crop: {crop_path}")

    attrs = detector.analyze(crop_path, actions=["age", "gender"])
    print("analyze() result (age/gender):")
    print(attrs)


def demo_batch_processing(
    detector: FaceDetectorDNN | FaceDetectorDeepFace,
    image_dir: Path,
    max_images: int,
) -> None:
    """Run detection over many images and print a summary."""
    print("\n" + "=" * 70)
    print("DEMO: Batch Processing")
    print("=" * 70)

    patterns = ("*.jpg", "*.jpeg", "*.png", "*.webp", "*.bmp", "*.tif", "*.tiff")
    paths: list[Path] = []
    for pat in patterns:
        paths.extend(p for p in image_dir.glob(pat) if p.is_file())

    paths = sorted({p.resolve() for p in paths})
    if max_images > 0:
        paths = paths[:max_images]

    image_path_args: list[str | Path] = list(paths)
    print(f"Batch input dir: {image_dir} | images: {len(paths)}")

    if isinstance(detector, FaceDetectorDNN):
        results = detector.detect_faces_batch(image_path_args, show_progress=True)
    else:
        # FaceDetectorDeepFace has no batch method — loop manually.
        results: dict[str, list[tuple[int, int, int, int, float]]] = {}
        for p in paths:
            try:
                results[str(p)] = detector.detect_face(p)
            except Exception as e:
                print(f"  Error on {p}: {e}")
                results[str(p)] = []

    total_faces = sum(len(v) for v in results.values())
    print("Batch summary:")
    print(f"  total_images: {len(results)}")
    print(f"  total_faces:  {total_faces}")


def demo_cache_speed(
    detector: FaceDetectorDNN | FaceDetectorDeepFace,
    image_path: Path,
) -> None:
    """Crude timing demo: uncached → cached call."""
    print("\n" + "=" * 70)
    print("DEMO: Cache Speed")
    print("=" * 70)

    if not isinstance(detector, FaceDetectorDNN):
        print("Cache not supported by this backend; skipping")
        return

    if detector.get_cache_size() == 0 and detector._cache is None:
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
    detector: FaceDetectorDNN | FaceDetectorDeepFace,
) -> None:
    """Demonstrate custom exceptions on invalid inputs."""
    print("\n" + "=" * 70)
    print("DEMO: Error Handling")
    print("=" * 70)

    try:
        detector.detect_face("nonexistent.jpg")
    except ImageLoadError as e:
        print(f"Caught ImageLoadError: {e}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main(argv: Iterable[str] | None = None) -> None:
    """CLI entrypoint."""
    parser = argparse.ArgumentParser(description="FaceDetectorDNN demo")

    parser.add_argument(
        "--preset",
        choices=["balanced", "fast", "accurate", "deepface", "custom"],
        default="balanced",
        help="Detector preset",
    )
    parser.add_argument(
        "--run",
        default="batch",
        help=(
            "Comma-separated demos: "
            "basic,metrics,visualize,extract,analyze,batch,cache,errors "
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
        help="Enable/disable caching",
    )
    parser.add_argument(
        "--max-cache-size",
        type=int,
        default=200,
        help="Maximum cache entries",
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
        "--require-cuda",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Fail fast if CUDA cannot be used",
    )

    # Custom-preset knobs
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

    args = parser.parse_args(list(argv) if argv is not None else None)

    logging.basicConfig(level=args.log_level, format="%(levelname)s - %(message)s")

    run_list = _parse_run_list(args.run)

    needs_image = any(
        n in run_list
        for n in ("basic", "metrics", "visualize", "extract", "analyze", "cache")
    )

    image_path = args.image
    if needs_image:
        if image_path is None:
            raise SystemExit(
                "--image is required for the selected demos "
                "(basic/metrics/visualize/extract/analyze/cache)"
            )
        if not image_path.exists():
            raise SystemExit(f"Image not found: {image_path}")

    detector = _build_detector(args)

    # Label for output filenames.
    cache_on = isinstance(detector, FaceDetectorDNN) and detector._cache is not None
    label = f"{args.preset}_cache_{'on' if cache_on else 'off'}"

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
    if "analyze" in run_list:
        demo_deepface_analyze_on_crop(detector, image_path, args.results_dir, label)
    if "batch" in run_list:
        demo_batch_processing(detector, args.batch_dir, args.max_images)
    if "cache" in run_list:
        demo_cache_speed(detector, image_path)
    if "errors" in run_list:
        demo_error_handling(detector)


if __name__ == "__main__":
    main()
