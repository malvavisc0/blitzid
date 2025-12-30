"""FaceDetectorDNN demonstration script.

This script showcases the public API exposed by [`framework/facade.py`](framework/facade.py:1)
via [`framework/__init__.py`](framework/__init__.py:1) and the presets from
[`framework/factory.py`](framework/factory.py:1).

It is intentionally a standalone script (not a pytest suite). Use
[`test_refactored.py`](test_refactored.py:1) for quick, assertion-based smoke tests.
"""

from __future__ import annotations

import argparse
import logging
import time
from pathlib import Path
from typing import Iterable

import cv2

from framework import FaceDetectorDNN, FaceDetectorFactory, ImageLoadError


def _parse_log_level(value: str) -> int:
    """Parse a log-level string (e.g. "INFO") into a `logging` level int."""
    value = value.upper()
    mapping = {
        "CRITICAL": logging.CRITICAL,
        "ERROR": logging.ERROR,
        "WARNING": logging.WARNING,
        "INFO": logging.INFO,
        "DEBUG": logging.DEBUG,
    }
    try:
        return mapping[value]
    except KeyError as e:
        raise argparse.ArgumentTypeError(
            f"Invalid log level: {value}. Choose from: {', '.join(mapping)}"
        ) from e


def _build_detector(args: argparse.Namespace) -> object:
    """Construct a detector instance based on CLI arguments.

    The demo supports multiple backends. The returned instance is expected to
    provide methods like `detect_face()`, `detect_face_with_metrics()`,
    `extract_faces()`, and `visualize_detections()`.
    """
    if args.preset == "fast":
        detector: object = FaceDetectorFactory.create_fast_detector(
            model_dir=args.model_dir,
            log_level=args.log_level,
            require_cuda=args.require_cuda,
        )
    elif args.preset == "accurate":
        detector = FaceDetectorFactory.create_accurate_detector(
            model_dir=args.model_dir,
            log_level=args.log_level,
            require_cuda=args.require_cuda,
        )
    elif args.preset == "balanced":
        detector = FaceDetectorFactory.create_balanced_detector(
            model_dir=args.model_dir,
            log_level=args.log_level,
            require_cuda=args.require_cuda,
        )
    elif args.preset == "deepface":
        detector = FaceDetectorFactory.create_deepface_detector(
            model_dir=args.model_dir,
            log_level=args.log_level,
            detector_backend="retinaface",
            align=True,
            enforce_detection=False,
        )

        # DeepFace uses TensorFlow/PyTorch depending on your install.
        # `--require-cuda` here means: require TF to see a GPU.
        if args.require_cuda:
            try:
                import tensorflow as tf  # type: ignore

                gpus = tf.config.list_physical_devices("GPU")
            except Exception as e:  # pragma: no cover
                raise SystemExit(
                    f"--require-cuda was set, but TensorFlow GPU is not available: {e}"
                )

            if not gpus:
                raise SystemExit(
                    "--require-cuda was set, but TensorFlow reports no visible GPU devices"
                )
    elif args.preset == "custom":
        detector = FaceDetectorDNN(
            confidence_threshold=args.confidence_threshold,
            min_face_size=tuple(args.min_face_size),
            nms_threshold=args.nms_threshold,
            log_level=args.log_level,
            enable_cache=args.cache,
            max_cache_size=args.max_cache_size,
            model_dir=args.model_dir,
            preprocess=args.preprocess,
            require_cuda=args.require_cuda,
        )
    else:  # pragma: no cover
        raise ValueError(f"Unknown preset: {args.preset}")

    # Override cache/preprocess only if the detector supports it.
    if hasattr(detector, "cache"):
        cache = getattr(detector, "cache")
        if hasattr(detector, "enable_cache"):
            setattr(detector, "enable_cache", args.cache)
        if hasattr(cache, "enabled"):
            setattr(cache, "enabled", args.cache)
        if hasattr(cache, "max_size"):
            setattr(cache, "max_size", args.max_cache_size)

    if hasattr(detector, "preprocess"):
        setattr(detector, "preprocess", args.preprocess)
    if hasattr(detector, "image_loader"):
        loader = getattr(detector, "image_loader")
        if hasattr(loader, "preprocess"):
            setattr(loader, "preprocess", args.preprocess)

    return detector


def demo_basic_detection(detector: object, image_path: Path) -> None:
    """Run a single detection call and print per-face results."""
    print("\n" + "=" * 70)
    print("DEMO: Basic Detection")
    print("=" * 70)

    faces = getattr(detector, "detect_face")(image_path)
    print(f"Image: {image_path} | faces: {len(faces)}")

    for idx, (x, y, w, h, conf) in enumerate(faces, 1):
        print(f"  Face {idx}: x={x} y={y} w={w} h={h} confidence={conf:.2%}")


def demo_metrics(detector: object, image_path: Path) -> None:
    """Show metrics output (miss -> hit when caching is enabled)."""
    print("\n" + "=" * 70)
    print("DEMO: Detection With Metrics")
    print("=" * 70)

    # Make the demo output meaningful: show a cache miss then a cache hit.
    if hasattr(detector, "cache") and getattr(
        getattr(detector, "cache"), "enabled", False
    ):
        if hasattr(detector, "clear_cache"):
            getattr(detector, "clear_cache")()

    r1 = getattr(detector, "detect_face_with_metrics")(image_path)
    r2 = getattr(detector, "detect_face_with_metrics")(image_path)

    print("First call:")
    print(f"  faces: {r1.num_faces}")
    print(f"  backend: {r1.backend}")
    print(f"  cache_hit: {r1.cache_hit}")
    print(f"  image_size: {r1.image_size}")
    print(f"  processing_time_ms: {r1.processing_time * 1000:.2f}")

    print("Second call:")
    print(f"  faces: {r2.num_faces}")
    print(f"  backend: {r2.backend}")
    print(f"  cache_hit: {r2.cache_hit}")
    print(f"  image_size: {r2.image_size}")
    print(f"  processing_time_ms: {r2.processing_time * 1000:.2f}")


def demo_visualization(
    detector: object,
    image_path: Path,
    results_dir: Path,
    label: str,
) -> None:
    """Draw detections and write visualization images into `results_dir`.

    Writes two outputs:
    - an "original" visualization (boxes drawn on raw image)
    - a "preprocessed" visualization (boxes drawn on the image after ImageLoader preprocessing)

    This makes it easy to spot over-aggressive preprocessing that alters appearance.
    """
    print("\n" + "=" * 70)
    print("DEMO: Visualization")
    print("=" * 70)

    results_dir.mkdir(parents=True, exist_ok=True)

    # Compute faces once (may involve preprocessing depending on detector settings).
    faces = getattr(detector, "detect_face")(image_path)

    # 1) Draw on the original pixels (no preprocessing) for sanity-checking visuals.
    raw = cv2.imread(str(image_path))
    if raw is None:
        raise SystemExit(f"Could not read image via OpenCV: {image_path}")

    out_path_original = results_dir / f"{label}_visualization_original.jpg"
    visualizer = getattr(detector, "visualizer")
    img_with_boxes_original = visualizer.draw_detections(
        raw,
        faces,
        show_confidence=True,
        color=(0, 255, 0),
        thickness=2,
    )
    cv2.imwrite(str(out_path_original), img_with_boxes_original)

    # 2) Draw on the potentially-preprocessed loader output (what the library returns today).
    out_path_preprocessed = (
        results_dir / f"{label}_visualization_preprocessed.jpg"
    )
    img_with_boxes_preprocessed = getattr(detector, "visualize_detections")(
        image_path,
        faces=faces,
        output_path=out_path_preprocessed,
        show_confidence=True,
        color=(0, 255, 0),
        thickness=2,
    )

    print(f"Saved visualization (original):     {out_path_original}")
    print(f"Saved visualization (preprocessed): {out_path_preprocessed}")
    print(f"Output image shape: {img_with_boxes_preprocessed.shape}")


def demo_face_extraction(
    detector: object,
    image_path: Path,
    results_dir: Path,
    label: str,
    padding: float,
) -> None:
    """Extract face crops and save them into `results_dir`."""
    print("\n" + "=" * 70)
    print("DEMO: Face Extraction")
    print("=" * 70)

    results_dir.mkdir(parents=True, exist_ok=True)
    extracted = getattr(detector, "extract_faces")(image_path, padding=padding)

    print(f"Extracted {len(extracted)} face crop(s)")

    for idx, (face_img, bbox, conf) in enumerate(extracted, 1):
        out_path = results_dir / f"{label}_face_{idx:03d}_conf_{conf:.2f}.jpg"
        ok = cv2.imwrite(str(out_path), face_img)
        print(
            f"  Face {idx}: bbox={bbox} conf={conf:.2%} shape={face_img.shape} "
            f"saved={ok} -> {out_path}"
        )


def demo_deepface_analyze_on_crop(
    detector: object,
    image_path: Path,
    results_dir: Path,
    label: str,
) -> None:
    """Demo: save a cropped face to disk and run `detector.analyze(path)`.

    This is intended for the DeepFace backend, but it will skip cleanly if the
    selected backend does not implement `analyze`.
    """
    print("\n" + "=" * 70)
    print("DEMO: DeepFace Analyze on Saved Crop")
    print("=" * 70)

    if not hasattr(detector, "analyze"):
        print("analyze() not supported by this backend; skipping")
        return

    results_dir.mkdir(parents=True, exist_ok=True)

    extracted = getattr(detector, "extract_faces")(image_path, padding=0.0)
    if not extracted:
        print("No faces extracted; skipping")
        return

    face_img, _bbox, conf = extracted[0]
    crop_path = results_dir / f"{label}_crop_for_analyze_conf_{conf:.2f}.jpg"
    ok = cv2.imwrite(str(crop_path), face_img)
    print(f"Wrote crop: {crop_path} (ok={ok})")

    # Run DeepFace attribute analysis on the saved crop path.
    attrs = getattr(detector, "analyze")(crop_path, actions=["age", "gender"])
    print("analyze() result (age/gender):")
    print(attrs)


def demo_batch_processing(
    detector: object,
    image_dir: Path,
    max_images: int,
) -> None:
    """Run detection over many images and print a summary.

    The demo intentionally processes *all* images in the directory (common
    extensions), so it can be used as a quick regression smoke test.
    """
    print("\n" + "=" * 70)
    print("DEMO: Batch Processing")
    print("=" * 70)

    patterns = (
        "*.jpg",
        "*.jpeg",
        "*.png",
        "*.webp",
        "*.bmp",
        "*.tif",
        "*.tiff",
    )
    paths: list[Path] = []
    for pat in patterns:
        paths.extend(p for p in image_dir.glob(pat) if p.is_file())

    # Deduplicate + stable order
    paths = sorted({p.resolve() for p in paths})

    if max_images > 0:
        paths = paths[:max_images]

    # Help type-checkers (List is invariant): explicitly widen the element type.
    image_path_args: list[str | Path] = list(paths)

    print(f"Batch input dir: {image_dir} | images: {len(paths)}")
    results = getattr(detector, "detect_faces_batch")(
        image_path_args, show_progress=True
    )

    total_faces = sum(len(v) for v in results.values())
    print("Batch summary:")
    print(f"  total_images: {len(results)}")
    print(f"  total_faces: {total_faces}")


def demo_cache_speed(detector: object, image_path: Path) -> None:
    """Crude timing demo for cache-on vs cached call."""
    print("\n" + "=" * 70)
    print("DEMO: Cache Speed")
    print("=" * 70)

    if not hasattr(detector, "cache"):
        print("Cache not supported by this backend; skipping")
        return

    cache = getattr(detector, "cache")
    if not getattr(cache, "enabled", False):
        print("Cache disabled; skipping")
        return

    start = time.time()
    _faces1 = getattr(detector, "detect_face")(image_path)
    t1 = time.time() - start

    start = time.time()
    _faces2 = getattr(detector, "detect_face")(image_path)
    t2 = time.time() - start

    print(f"First detection:  {t1 * 1000:.2f} ms")
    print(f"Second (cached):  {t2 * 1000:.2f} ms")
    if t2 > 0:
        print(f"Speedup:         {t1 / t2:.2f}x")


def demo_error_handling(detector: object) -> None:
    """Demonstrate the framework's custom exceptions on invalid inputs."""
    print("\n" + "=" * 70)
    print("DEMO: Error Handling")
    print("=" * 70)

    try:
        getattr(detector, "detect_face")("nonexistent.jpg")
    except ImageLoadError as e:
        print(f"Caught ImageLoadError: {e}")


def _parse_run_list(value: str) -> list[str]:
    """Parse the `--run` CLI argument into a list of demo names."""
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

    items = [v.strip().lower() for v in value.split(",") if v.strip()]
    return items


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
        help="Comma-separated demos: basic,metrics,visualize,extract,analyze,batch,cache,errors or 'all'",
    )
    parser.add_argument(
        "--image",
        type=Path,
        default=None,
        help="Path to a test image (required for basic/metrics/visualize/extract/analyze/cache demos)",
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
        help="Model directory (default is repo-root/models)",
    )

    parser.add_argument(
        "--preprocess",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Enable/disable basic preprocessing (CLAHE on luminance)",
    )

    parser.add_argument(
        "--require-cuda",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Fail fast if CUDA cannot be used. For DNN presets, this requires OpenCV DNN CUDA. "
            "For the deepface preset, this requires TensorFlow to report at least one GPU."
        ),
    )

    # Custom preset knobs
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

    # The demo configures logging for visibility; the library itself does not add handlers.
    logging.basicConfig(
        level=args.log_level, format="%(levelname)s - %(message)s"
    )

    run_list = _parse_run_list(args.run)

    # `--image` is intentionally optional so the default run can be batch-only.
    needs_image = any(
        name in run_list
        for name in (
            "basic",
            "metrics",
            "visualize",
            "extract",
            "analyze",
            "cache",
        )
    )

    image_path = args.image
    if needs_image:
        if image_path is None:
            raise SystemExit(
                "--image is required for the selected demos (basic/metrics/visualize/extract/analyze/cache)"
            )
        if not image_path.exists():
            raise SystemExit(f"Image not found: {image_path}")

    detector = _build_detector(args)

    cache_on = bool(
        getattr(getattr(detector, "cache", object()), "enabled", False)
    )
    label = f"{args.preset}_cache_{'on' if cache_on else 'off'}"

    if "basic" in run_list:
        demo_basic_detection(detector, image_path)

    if "metrics" in run_list:
        demo_metrics(detector, image_path)

    if "visualize" in run_list:
        demo_visualization(detector, image_path, args.results_dir, label)

    if "extract" in run_list:
        demo_face_extraction(
            detector,
            image_path,
            args.results_dir,
            label,
            padding=args.padding,
        )

    if "analyze" in run_list:
        demo_deepface_analyze_on_crop(
            detector, image_path, args.results_dir, label
        )

    if "batch" in run_list:
        demo_batch_processing(detector, args.batch_dir, args.max_images)

    if "cache" in run_list:
        demo_cache_speed(detector, image_path)

    if "errors" in run_list:
        demo_error_handling(detector)

    # Metrics tracker summary (only `detect_face_with_metrics` records into it).
    # Print this only when the metrics demo was requested; otherwise it is noisy
    # and usually contains only zeros.
    if "metrics" in run_list and hasattr(detector, "metrics"):
        stats = getattr(detector, "metrics").get_stats()
        print("\n" + "=" * 70)
        print("MetricsTracker summary (from detect_face_with_metrics calls)")
        print("=" * 70)
        for k, v in stats.items():
            print(f"  {k}: {v}")


if __name__ == "__main__":
    main()
