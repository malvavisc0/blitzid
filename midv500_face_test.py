from __future__ import annotations

import argparse
import statistics
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import cv2

from framework import FaceDetectorDNN
from framework.exceptions import FaceDetectorError


@dataclass(frozen=True)
class RunStats:
    images_total: int
    images_processed: int
    images_ok: int
    images_failed: int
    faces_total: int


def _find_set_roots(root: Path) -> list[Path]:
    """Return set roots under `root`.

    Supports inputs:
    - a parent folder containing `midv500/` and/or `midv2019/`
    - a set folder itself (e.g. `.../midv500/`)
    - a document folder (contains `images/` and `ground_truth/`)

    The return value is a list of folders that contain one or more document roots.
    """

    # Document root passed
    if (root / "images").is_dir():
        return [root]

    # Set root passed
    if any((root / d).is_dir() for d in ("01_alb_id", "02_aut_drvlic_new")):
        return [root]

    # Parent root passed
    sets: list[Path] = []
    for name in ("midv500", "midv2019"):
        p = root / name
        if p.is_dir():
            sets.append(p)

    # Fallback: treat `root` as a set-root-like folder
    return sets or [root]


def _find_document_roots(set_root: Path) -> list[Path]:
    if (set_root / "images").is_dir() and (set_root / "ground_truth").is_dir():
        return [set_root]

    out: list[Path] = []
    for p in sorted(set_root.iterdir()):
        if not p.is_dir():
            continue
        if (p / "images").is_dir() and (p / "ground_truth").is_dir():
            out.append(p)
    return out


def _iter_image_paths(doc_root: Path, include_root_images: bool) -> list[Path]:
    images_root = doc_root / "images"
    exts = {".tif", ".tiff", ".png", ".jpg", ".jpeg", ".webp", ".bmp"}
    paths: list[Path] = []

    for p in images_root.rglob("*"):
        if not p.is_file():
            continue
        if p.suffix.lower() not in exts:
            continue
        if not include_root_images and p.relative_to(images_root).parent == Path("."):
            # Skip top-level reference images like `images/<docname>.tif`.
            continue
        paths.append(p)

    return sorted(paths)


def _save_visualization(
    detector: FaceDetectorDNN,
    image_path: Path,
    faces: list[tuple[int, int, int, int, float]],
    out_dir: Path,
    tag: str,
) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    safe_tag = "".join(c if c.isalnum() or c in ("-", "_", ".") else "_" for c in tag)
    out_path = out_dir / f"{safe_tag}.jpg"

    detector.visualize_detections(
        image_path,
        faces=faces,
        output_path=out_path,
        show_confidence=True,
        color=(0, 255, 0),
        thickness=2,
    )
    return out_path


def _save_face_crops(
    detector: FaceDetectorDNN,
    image_path: Path,
    out_dir: Path,
    doc_name: str,
    max_faces_per_image: int,
    face_padding: float,
    global_saved: int,
    max_saved_total: int,
) -> tuple[int, list[tuple[int, int, int, int, float]]]:
    """Extract face crops, save them to disk, and return (saved_count, faces).

    Returns:
        saved_count: number of crops saved for this image
        faces: list of detections in (x, y, w, h, conf) format
    """

    extracted = detector.extract_faces(image_path, padding=face_padding)
    faces = [(x, y, w, h, conf) for (_img, (x, y, w, h), conf) in extracted]

    # Apply per-image limit after extracting.
    if max_faces_per_image > 0:
        extracted = extracted[:max_faces_per_image]

    saved = 0
    for idx, (face_img, _bbox, conf) in enumerate(extracted, 1):
        if max_saved_total > 0 and global_saved + saved >= max_saved_total:
            break

        # Keep the layout human-browsable.
        crop_dir = out_dir / doc_name
        crop_dir.mkdir(parents=True, exist_ok=True)

        out_path = crop_dir / (
            f"{image_path.stem}_face_{idx:02d}_conf_{conf:.3f}.jpg"
        )
        ok = cv2.imwrite(str(out_path), face_img)
        if ok:
            saved += 1

    return saved, faces


def run(
    root: Path,
    max_docs: int,
    max_images_per_doc: int,
    max_images_total: int,
    save_vis: int,
    results_dir: Path | None,
    save_faces: bool,
    faces_dir: Path,
    max_saved_faces_total: int,
    max_faces_per_image: int,
    face_padding: float,
    include_root_images: bool,
    log_level: str,
) -> int:
    try:
        detector = FaceDetectorDNN(
            log_level=_parse_log_level(log_level),
            preprocess=True,
        )
    except Exception as e:
        print(
            "Failed to initialize FaceDetectorDNN (model download/load may have failed): "
            f"{e}"
        )
        return 2

    set_roots = _find_set_roots(root)

    doc_roots: list[Path] = []
    for set_root in set_roots:
        doc_roots.extend(_find_document_roots(set_root))

    if not doc_roots:
        print(
            f"No MIDV-500 document roots found under: {root}\n"
            "Expected folders containing 'images/' and 'ground_truth/'."
        )
        return 2

    if max_docs > 0:
        doc_roots = doc_roots[:max_docs]

    failures: list[tuple[Path, str]] = []
    faces_per_image: list[int] = []
    vis_saved = 0
    faces_saved_total = 0

    processed = 0
    ok = 0

    for doc_root in doc_roots:
        image_paths = _iter_image_paths(doc_root, include_root_images=include_root_images)
        if max_images_per_doc > 0:
            image_paths = image_paths[:max_images_per_doc]

        print(f"Doc: {doc_root.name} | images_selected={len(image_paths)}")

        for image_path in image_paths:
            if max_images_total > 0 and processed >= max_images_total:
                break

            processed += 1

            try:
                if save_faces:
                    saved_here, faces = _save_face_crops(
                        detector,
                        image_path=image_path,
                        out_dir=faces_dir,
                        doc_name=doc_root.name,
                        max_faces_per_image=max_faces_per_image,
                        face_padding=face_padding,
                        global_saved=faces_saved_total,
                        max_saved_total=max_saved_faces_total,
                    )
                    faces_saved_total += saved_here
                else:
                    faces = detector.detect_face(image_path)

                ok += 1
                faces_per_image.append(len(faces))

                if results_dir is not None and vis_saved < save_vis and faces:
                    tag = f"{doc_root.name}_{image_path.stem}_faces_{len(faces)}"
                    out_path = _save_visualization(
                        detector,
                        image_path=image_path,
                        faces=faces,
                        out_dir=results_dir,
                        tag=tag,
                    )
                    vis_saved += 1
                    print(f"  saved_vis: {out_path}")

            except FaceDetectorError as e:
                failures.append((image_path, str(e)))
            except Exception as e:
                failures.append((image_path, f"Unexpected error: {e}"))

        if max_images_total > 0 and processed >= max_images_total:
            break

    failed = len(failures)
    faces_total = sum(faces_per_image)

    print("\nSummary:")
    print(f"  set_roots:           {len(set_roots)}")
    print(f"  doc_roots:           {len(doc_roots)}")
    print(f"  images_processed:    {processed}")
    print(f"  ok:                 {ok}")
    print(f"  failed:             {failed}")
    print(f"  faces_total:        {faces_total}")
    print(f"  vis_saved:          {vis_saved}")
    if save_faces:
        print(f"  faces_saved_total:  {faces_saved_total} -> {faces_dir}")

    if faces_per_image:
        print("  faces/image stats:")
        print(f"    mean:   {statistics.mean(faces_per_image):.3f}")
        print(f"    median: {statistics.median(faces_per_image):.3f}")
        print(f"    max:    {max(faces_per_image)}")
        print(f"    min:    {min(faces_per_image)}")

    if failures:
        print("\nFirst failures:")
        for p, msg in failures[:10]:
            print(f"- {p}: {msg}")

    return 0 if failed == 0 else 1


def _parse_log_level(value: str) -> int:
    import logging

    value = value.strip().upper()
    mapping = {
        "CRITICAL": logging.CRITICAL,
        "ERROR": logging.ERROR,
        "WARNING": logging.WARNING,
        "INFO": logging.INFO,
        "DEBUG": logging.DEBUG,
    }
    if value not in mapping:
        raise argparse.ArgumentTypeError(
            f"Invalid log level: {value}. Choose from: {', '.join(mapping)}"
        )
    return mapping[value]


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Run FaceDetectorDNN on MIDV-500 images")
    p.add_argument(
        "root",
        nargs="?",
        default="data/midv500",
        help=(
            "Dataset root. Can be a folder containing midv500/ and midv2019/, "
            "or a set root, or a single document root."
        ),
    )

    p.add_argument("--max-docs", type=int, default=3, help="Limit document folders (0 = all).")
    p.add_argument(
        "--max-images-per-doc",
        type=int,
        default=20,
        help="Limit images tested per document folder (0 = all).",
    )
    p.add_argument(
        "--max-images-total",
        type=int,
        default=100,
        help="Global cap across all docs (0 = no cap).",
    )

    p.add_argument(
        "--results-dir",
        type=Path,
        default=Path("results/midv500_face_test"),
        help="Where to save optional visualizations.",
    )
    p.add_argument(
        "--save-vis",
        type=int,
        default=5,
        help="Save up to N visualization images (0 = disable saving).",
    )

    p.add_argument(
        "--save-faces",
        action="store_true",
        help="Extract detected face crops and save them to disk (JPEG).",
    )
    p.add_argument(
        "--faces-dir",
        type=Path,
        default=Path("results/midv500_face_crops"),
        help="Output directory for extracted face crops (used with --save-faces).",
    )
    p.add_argument(
        "--max-saved-faces-total",
        type=int,
        default=200,
        help="Cap on the number of crops written across the whole run (0 = no cap).",
    )
    p.add_argument(
        "--max-faces-per-image",
        type=int,
        default=5,
        help="Cap on crops written per image (0 = no cap).",
    )
    p.add_argument(
        "--face-padding",
        type=float,
        default=0.2,
        help="Padding fraction used by extract_faces() (0.0-1.0).",
    )

    p.add_argument(
        "--include-root-images",
        action="store_true",
        help="Also test top-level reference images like images/<docname>.tif.",
    )

    p.add_argument(
        "--log-level",
        default="WARNING",
        help="Detector log level: DEBUG, INFO, WARNING, ERROR.",
    )

    return p


def main(argv: Iterable[str] | None = None) -> None:
    args = _build_parser().parse_args(list(argv) if argv is not None else None)

    results_dir = None if args.save_vis <= 0 else Path(args.results_dir)

    code = run(
        root=Path(args.root),
        max_docs=int(args.max_docs),
        max_images_per_doc=int(args.max_images_per_doc),
        max_images_total=int(args.max_images_total),
        save_vis=int(args.save_vis),
        results_dir=results_dir,
        save_faces=bool(args.save_faces),
        faces_dir=Path(args.faces_dir),
        max_saved_faces_total=int(args.max_saved_faces_total),
        max_faces_per_image=int(args.max_faces_per_image),
        face_padding=float(args.face_padding),
        include_root_images=bool(args.include_root_images),
        log_level=str(args.log_level),
    )

    raise SystemExit(code)


if __name__ == "__main__":
    main()
