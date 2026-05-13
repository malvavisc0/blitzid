"""CUDA diagnostics and CPU-vs-GPU benchmark for BlitzID.

Reports:
- OpenCV build configuration (CUDA flags)
- Available CUDA devices (via OpenCV and optionally TensorFlow)
- Face detection speed comparison: CPU baseline vs CUDA backend

Run::

    python scripts/cuda_diagnostics.py
    python scripts/cuda_diagnostics.py --image images/bub_der_personalausweis_kopie.jpg --rounds 20
    python scripts/cuda_diagnostics.py --check-tensorflow
"""

from __future__ import annotations

import argparse
import statistics
import sys
import time
from pathlib import Path

import cv2
import numpy as np

# ---------------------------------------------------------------------------
# Diagnostics
# ---------------------------------------------------------------------------


def _section(title: str) -> None:
    print(f"\n{'=' * 70}")
    print(f"  {title}")
    print("=" * 70)


def report_opencv_build() -> dict[str, str]:
    """Parse and display CUDA-related lines from OpenCV build info."""
    _section("OpenCV Build Info")
    print(f"  Version:  {cv2.__version__}")

    info = cv2.getBuildInformation()
    cuda_lines: dict[str, str] = {}
    keywords = ("NVIDIA CUDA", "CUDA", "cuDNN", "CUFFT", "CUBLAS", "NVCUVID")

    for line in info.splitlines():
        stripped = line.strip()
        for kw in keywords:
            if stripped.startswith(kw):
                key, _, val = stripped.partition(":")
                cuda_lines[key.strip()] = val.strip()

    if cuda_lines:
        for k, v in cuda_lines.items():
            print(f"  {k}: {v}")
    else:
        print("  No CUDA-related build flags found — OpenCV was built without CUDA.")

    return cuda_lines


def report_cuda_devices() -> int:
    """Report CUDA devices visible to OpenCV."""
    _section("OpenCV CUDA Devices")

    try:
        count = cv2.cuda.getCudaEnabledDeviceCount()
    except Exception as e:
        print(f"  cv2.cuda module unavailable: {e}")
        return 0

    print(f"  Enabled device count: {count}")

    for i in range(count):
        try:
            cv2.cuda.setDevice(i)
            dev = cv2.cuda.getDevice()
            print(f"\n  Device {i} (active={dev == i}):")

            try:
                info = cv2.cuda.DeviceInfo(i)  # type: ignore[attr-defined]
                for attr in ("name", "totalMemory", "majorVersion", "minorVersion"):
                    if hasattr(info, attr):
                        val = getattr(info, attr)
                        val = val() if callable(val) else val
                        if attr == "totalMemory":
                            val = f"{val / (1024**2):.0f} MB"
                        print(f"    {attr}: {val}")
            except (AttributeError, Exception):
                try:
                    cv2.cuda.printCudaDeviceInfo(i)  # type: ignore[attr-defined]
                except (AttributeError, Exception):
                    print("    (device info not available in this build)")

        except cv2.error as e:
            print(f"    Error querying device {i}: {e}")

    return count


def report_tensorflow_gpu() -> None:
    """Optionally report TensorFlow GPU visibility (for DeepFace users)."""
    _section("TensorFlow GPU (optional)")

    try:
        import tensorflow as tf  # type: ignore[import-untyped]
    except ImportError:
        print("  TensorFlow not installed — skipping.")
        return

    print(f"  TensorFlow version: {tf.__version__}")

    gpus = tf.config.list_physical_devices("GPU")
    print(f"  Visible GPU devices: {len(gpus)}")
    for g in gpus:
        print(f"    {g}")

    if not gpus:
        print("  No GPU visible to TensorFlow.")


# ---------------------------------------------------------------------------
# Benchmark
# ---------------------------------------------------------------------------


def _load_or_generate_image(image_path: Path | None) -> np.ndarray:
    """Load a real image or generate a synthetic 640×480 test image."""
    if image_path and image_path.exists():
        img = cv2.imread(str(image_path))
        if img is not None:
            return img
        print(f"  Warning: could not read {image_path}, using synthetic image")

    # Synthetic image with some structure (gradient + noise) so the DNN
    # doesn't just short-circuit on a blank frame.
    rng = np.random.default_rng(42)
    img = rng.integers(0, 256, (480, 640, 3), dtype=np.uint8)
    return img


def run_benchmark(
    image: np.ndarray,
    rounds: int,
    model_dir: Path | None,
) -> None:
    """Run CPU baseline and (if available) CUDA benchmark."""
    from blitzid._models import ModelManager, default_model_dir

    _section("Benchmark: CPU vs CUDA")

    mdir = model_dir or default_model_dir()
    import logging

    logger = logging.getLogger("cuda_diag")
    logger.setLevel(logging.WARNING)
    mgr = ModelManager(mdir, logger)

    # --- CPU run ---
    print("\n  [CPU] Loading network...")
    try:
        net_cpu, _ = mgr.load_network(use_cuda=False)
    except Exception as e:
        print(f"  CPU load failed: {e}")
        return

    times_cpu = _bench_forward(net_cpu, image, rounds)
    _print_stats("CPU", times_cpu)

    # --- CUDA run ---
    cuda_available = ModelManager._opencv_has_cuda_support()
    if not cuda_available:
        print("\n  [CUDA] Skipped — OpenCV built without CUDA support.")
        return

    try:
        cuda_count = cv2.cuda.getCudaEnabledDeviceCount()
    except Exception:
        cuda_count = 0

    if cuda_count == 0:
        print("\n  [CUDA] Skipped — no CUDA devices visible.")
        return

    print(f"\n  [CUDA] Loading network (devices={cuda_count})...")
    try:
        net_cuda, backend = mgr.load_network(use_cuda=True, require_cuda=True)
    except Exception as e:
        print(f"  CUDA load failed: {e}")
        return

    print(f"  [CUDA] Backend: {backend}")

    # Warm-up pass (first CUDA inference is slow due to kernel compilation).
    _forward_pass(net_cuda, image)

    times_cuda = _bench_forward(net_cuda, image, rounds)
    _print_stats("CUDA", times_cuda)

    # --- Comparison ---
    if times_cpu and times_cuda:
        mean_cpu = statistics.mean(times_cpu)
        mean_cuda = statistics.mean(times_cuda)
        speedup = mean_cpu / mean_cuda if mean_cuda > 0 else float("inf")
        print(f"\n  Speedup (mean): {speedup:.2f}x")


def _forward_pass(net: cv2.dnn.Net, image: np.ndarray) -> None:
    """Single DNN forward pass."""
    blob = cv2.dnn.blobFromImage(
        image=image,
        scalefactor=1.0,
        size=(300, 300),
        mean=(104.0, 177.0, 123.0),
        swapRB=False,
        crop=False,
    )
    net.setInput(blob)
    net.forward()


def _bench_forward(
    net: cv2.dnn.Net,
    image: np.ndarray,
    rounds: int,
) -> list[float]:
    """Time *rounds* forward passes, return per-round durations in ms."""
    times: list[float] = []
    for _ in range(rounds):
        start = time.perf_counter()
        _forward_pass(net, image)
        elapsed_ms = (time.perf_counter() - start) * 1000.0
        times.append(elapsed_ms)
    return times


def _print_stats(label: str, times: list[float]) -> None:
    if not times:
        return
    print(f"\n  [{label}] {len(times)} rounds:")
    print(f"    mean:   {statistics.mean(times):8.2f} ms")
    print(f"    median: {statistics.median(times):8.2f} ms")
    print(f"    min:    {min(times):8.2f} ms")
    print(f"    max:    {max(times):8.2f} ms")
    if len(times) > 1:
        print(f"    stdev:  {statistics.stdev(times):8.2f} ms")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description="CUDA diagnostics and CPU-vs-GPU benchmark for BlitzID"
    )
    parser.add_argument(
        "--image",
        type=Path,
        default=None,
        help="Image to benchmark (default: synthetic 640×480)",
    )
    parser.add_argument(
        "--rounds",
        type=int,
        default=10,
        help="Number of forward-pass rounds (default: 10)",
    )
    parser.add_argument(
        "--model-dir",
        type=Path,
        default=None,
        help="Model directory override",
    )
    parser.add_argument(
        "--check-tensorflow",
        action="store_true",
        help="Also check TensorFlow GPU visibility",
    )
    parser.add_argument(
        "--skip-benchmark",
        action="store_true",
        help="Only run diagnostics, skip benchmark",
    )

    args = parser.parse_args()

    # --- Diagnostics ---
    cuda_lines = report_opencv_build()
    device_count = report_cuda_devices()

    if args.check_tensorflow:
        report_tensorflow_gpu()

    # --- Summary ---
    _section("Summary")
    has_cuda_build = any("YES" in v.upper() for v in cuda_lines.values())
    print(f"  OpenCV CUDA build:  {'Yes' if has_cuda_build else 'No'}")
    print(f"  CUDA devices:       {device_count}")

    if not has_cuda_build:
        print("\n  ⚠ OpenCV was built without CUDA. To use GPU acceleration,")
        print(
            "    rebuild OpenCV with -DWITH_CUDA=ON or install opencv-contrib-python-cuda."
        )

    # --- Benchmark ---
    if args.skip_benchmark:
        print("\n  Benchmark skipped (--skip-benchmark).")
        return

    image = _load_or_generate_image(args.image)
    print(
        f"\n  Benchmark image: {image.shape[1]}×{image.shape[0]} "
        f"({'file' if args.image else 'synthetic'})"
    )

    run_benchmark(image, rounds=args.rounds, model_dir=args.model_dir)


if __name__ == "__main__":
    main()
