"""Benchmark script — timing for detection, cache, and OCR/MRZ pipelines.

Run examples::

    python scripts/benchmark.py
    python scripts/benchmark.py --runs 30 --markdown

Results are printed and written as markdown into the results directory
(default: ``results/``, a local artifact — not committed).
"""

from __future__ import annotations

import argparse
import logging
import platform
import statistics
import time
from collections.abc import Callable, Iterable
from pathlib import Path

import onnxruntime as ort  # type: ignore[import-untyped]

from blitzid import FaceDetectorDNN

PRESETS = ("fast", "balanced", "accurate")


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark blitzid pipelines.")
    parser.add_argument(
        "--image",
        type=Path,
        default=Path("images/bub_der_personalausweis_kopie.jpg"),
        help="Image for the detection benchmarks (path).",
    )
    parser.add_argument(
        "--mrz-image",
        type=Path,
        default=Path("images/nl_td1_id_specimen.jpg"),
        help="Document image for the OCR/MRZ benchmarks (path).",
    )
    parser.add_argument(
        "--runs",
        type=int,
        default=20,
        help="Timed runs per benchmark (default: 20).",
    )
    parser.add_argument(
        "--markdown",
        action="store_true",
        help="Print results as a markdown table.",
    )
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=Path("results"),
        help="Directory to write the benchmark markdown file into.",
    )
    return parser.parse_args(argv)


def _timed_ms(func: Callable[[], object], runs: int) -> list[float]:
    """Return *runs* wall-clock timings in ms after one warm-up call."""
    func()
    timings = []
    for _ in range(runs):
        start = time.perf_counter()
        func()
        timings.append((time.perf_counter() - start) * 1000)
    return timings


def _detector(preset: str) -> FaceDetectorDNN:
    factory = {
        "fast": FaceDetectorDNN.create_fast_detector,
        "balanced": FaceDetectorDNN.create_balanced_detector,
        "accurate": FaceDetectorDNN.create_accurate_detector,
    }[preset]
    return factory(log_level=logging.WARNING)


def _benchmark_detection(
    preset: str, image: Path, runs: int
) -> tuple[str, str, float, float]:
    """Time engine init and warm detection for one preset."""
    start = time.perf_counter()
    detector = _detector(preset)
    init_ms = (time.perf_counter() - start) * 1000

    timings = _timed_ms(lambda: detector.detect_face(image), runs)
    return (
        preset,
        f"detect_face ({image.name})",
        init_ms,
        statistics.median(timings),
    )


def _benchmark_cache(image: Path, runs: int) -> tuple[str, str, float, float]:
    """Time cache miss vs hit via detect_face_with_metrics."""
    detector = FaceDetectorDNN.create_balanced_detector(log_level=logging.WARNING)
    detector.detect_face_with_metrics(image)  # prime the cache
    timings = _timed_ms(lambda: detector.detect_face_with_metrics(image), runs)
    return (
        "balanced",
        "detect_face_with_metrics (cache hit)",
        0.0,
        statistics.median(timings),
    )


def _benchmark_reading(
    name: str, func: Callable[[], object], runs: int
) -> tuple[str, str, float, float]:
    """Time an OCR/MRZ read; init is part of the first warm-up call."""
    start = time.perf_counter()
    func()
    init_ms = (time.perf_counter() - start) * 1000
    timings = _timed_ms(func, runs)
    return (name, "read", init_ms, statistics.median(timings))


def _benchmark_ocr(mrz_image: Path, runs: int) -> list[tuple[str, str, float, float]]:
    """Benchmark RapidOCRReader and MRZReader; skipped without the extra."""
    try:
        import rapidocr  # noqa: F401
    except ImportError:
        print("(ocr benchmarks skipped: rapidocr not installed)")
        return []

    from blitzid import MRZReader, RapidOCRReader

    reader = RapidOCRReader(log_level=logging.WARNING)
    mrz = MRZReader(reader=reader, log_level=logging.WARNING)
    return [
        _benchmark_reading("RapidOCRReader", lambda: reader.read(mrz_image), runs),
        _benchmark_reading("MRZReader", lambda: mrz.read(mrz_image), runs),
    ]


def _print_header(markdown: bool) -> None:
    print(
        f"python {platform.python_version()} | "
        f"onnxruntime {ort.__version__} | {platform.machine()}"
    )
    if markdown:
        print("\n| Pipeline | Benchmark | Init (ms) | Median (ms) |")
        print("|---|---|---|---|")


def _print_row(row: tuple[str, str, float, float], markdown: bool) -> None:
    pipeline, name, init_ms, median_ms = row
    if markdown:
        print(f"| {pipeline} | {name} | {init_ms:.1f} | {median_ms:.1f} |")
    else:
        print(
            f"{pipeline:<14} {name:<45} "
            f"init {init_ms:>8.1f} ms   median {median_ms:>8.1f} ms"
        )


def _table_rows(rows: list[tuple[str, str, float, float]]) -> list[str]:
    """Render the environment line and results as markdown lines."""
    lines = [
        f"python {platform.python_version()} | "
        f"onnxruntime {ort.__version__} | {platform.machine()}",
        "",
        "| Pipeline | Benchmark | Init (ms) | Median (ms) |",
        "|---|---|---|---|",
    ]
    lines.extend(
        f"| {pipeline} | {name} | {init_ms:.1f} | {median_ms:.1f} |"
        for pipeline, name, init_ms, median_ms in rows
    )
    return lines


def _write_results(
    rows: list[tuple[str, str, float, float]], results_dir: Path
) -> Path:
    """Write the benchmark table to a markdown file; return its path."""
    results_dir.mkdir(parents=True, exist_ok=True)
    stamp = time.strftime("%Y%m%d-%H%M%S")
    out_path = results_dir / f"benchmark-{stamp}.md"
    out_path.write_text("\n".join(_table_rows(rows)) + "\n")
    return out_path


def main(argv: Iterable[str] | None = None) -> None:
    """CLI entrypoint."""
    args = _parse_args(list(argv) if argv is not None else None)
    logging.basicConfig(level=logging.WARNING)

    rows = [_benchmark_detection(preset, args.image, args.runs) for preset in PRESETS]
    rows.append(_benchmark_cache(args.image, args.runs))
    rows.extend(_benchmark_ocr(args.mrz_image, args.runs))

    _print_header(args.markdown)
    for row in rows:
        _print_row(row, args.markdown)

    out_path = _write_results(rows, args.results_dir)
    print(f"\nResults written to {out_path}")


if __name__ == "__main__":
    main()
