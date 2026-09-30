"""MIDV-500 / MIDV-2019 dataset tooling.

Download the dataset and run FaceDetectorDNN over it.

Run examples::

    python scripts/midv500.py download data/
    python scripts/midv500.py download data/ --dataset all --keep-zip
    python scripts/midv500.py test data/midv500 --max-docs 3
    python scripts/midv500.py test data/ --save-faces --max-images-total 50
"""

from __future__ import annotations

import argparse
import statistics
import zipfile
from collections.abc import Iterable
from dataclasses import dataclass, field
from pathlib import Path
from urllib.request import urlretrieve

import cv2
from tqdm import tqdm

from blitzid import FaceDetectorDNN
from blitzid.exceptions import BlitzIDError

# ── Download ─────────────────────────────────────────────────


class TqdmUpTo(tqdm):  # type: ignore[type-arg]
    """`tqdm` progress bar compatible with `urllib.request.urlretrieve()` hooks."""

    def update_to(self, b: int = 1, bsize: int = 1, tsize: int | None = None) -> None:
        """Update the progress bar with `b` blocks of size `bsize`."""
        if tsize is not None:
            self.total = tsize
        self.update(b * bsize - self.n)


def download(url: str, save_dir: str | Path) -> Path:
    """Download a file to `save_dir` and return the local path."""

    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    filename = url.split("/")[-1]
    out_path = save_dir / filename

    with TqdmUpTo(unit="B", unit_scale=True, miniters=1, desc=filename) as t:
        urlretrieve(url, filename=str(out_path), reporthook=t.update_to, data=None)

    return out_path


def unzip(file_path: str | Path, dest_dir: str | Path) -> None:
    """Unzip `file_path` into `dest_dir`."""

    with zipfile.ZipFile(str(file_path)) as zf:
        zf.extractall(str(dest_dir))


midv500_links: list[str] = [
    "ftp://smartengines.com/midv-500/dataset/01_alb_id.zip",
    "ftp://smartengines.com/midv-500/dataset/02_aut_drvlic_new.zip",
    "ftp://smartengines.com/midv-500/dataset/03_aut_id_old.zip",
    "ftp://smartengines.com/midv-500/dataset/04_aut_id.zip",
    "ftp://smartengines.com/midv-500/dataset/05_aze_passport.zip",
    "ftp://smartengines.com/midv-500/dataset/06_bra_passport.zip",
    "ftp://smartengines.com/midv-500/dataset/07_chl_id.zip",
    "ftp://smartengines.com/midv-500/dataset/08_chn_homereturn.zip",
    "ftp://smartengines.com/midv-500/dataset/09_chn_id.zip",
    "ftp://smartengines.com/midv-500/dataset/10_cze_id.zip",
    "ftp://smartengines.com/midv-500/dataset/11_cze_passport.zip",
    "ftp://smartengines.com/midv-500/dataset/12_deu_drvlic_new.zip",
    "ftp://smartengines.com/midv-500/dataset/13_deu_drvlic_old.zip",
    "ftp://smartengines.com/midv-500/dataset/14_deu_id_new.zip",
    "ftp://smartengines.com/midv-500/dataset/15_deu_id_old.zip",
    "ftp://smartengines.com/midv-500/dataset/16_deu_passport_new.zip",
    "ftp://smartengines.com/midv-500/dataset/17_deu_passport_old.zip",
    "ftp://smartengines.com/midv-500/dataset/18_dza_passport.zip",
    "ftp://smartengines.com/midv-500/dataset/19_esp_drvlic.zip",
    "ftp://smartengines.com/midv-500/dataset/20_esp_id_new.zip",
    "ftp://smartengines.com/midv-500/dataset/21_esp_id_old.zip",
    "ftp://smartengines.com/midv-500/dataset/22_est_id.zip",
    "ftp://smartengines.com/midv-500/dataset/23_fin_drvlic.zip",
    "ftp://smartengines.com/midv-500/dataset/24_fin_id.zip",
    "ftp://smartengines.com/midv-500/dataset/25_grc_passport.zip",
    "ftp://smartengines.com/midv-500/dataset/26_hrv_drvlic.zip",
    "ftp://smartengines.com/midv-500/dataset/27_hrv_passport.zip",
    "ftp://smartengines.com/midv-500/dataset/28_hun_passport.zip",
    "ftp://smartengines.com/midv-500/dataset/29_irn_drvlic.zip",
    "ftp://smartengines.com/midv-500/dataset/30_ita_drvlic.zip",
    "ftp://smartengines.com/midv-500/dataset/31_jpn_drvlic.zip",
    "ftp://smartengines.com/midv-500/dataset/32_lva_passport.zip",
    "ftp://smartengines.com/midv-500/dataset/33_mac_id.zip",
    "ftp://smartengines.com/midv-500/dataset/34_mda_passport.zip",
    "ftp://smartengines.com/midv-500/dataset/35_nor_drvlic.zip",
    "ftp://smartengines.com/midv-500/dataset/36_pol_drvlic.zip",
    "ftp://smartengines.com/midv-500/dataset/37_prt_id.zip",
    "ftp://smartengines.com/midv-500/dataset/38_rou_drvlic.zip",
    "ftp://smartengines.com/midv-500/dataset/39_rus_internalpassport.zip",
    "ftp://smartengines.com/midv-500/dataset/40_srb_id.zip",
    "ftp://smartengines.com/midv-500/dataset/41_srb_passport.zip",
    "ftp://smartengines.com/midv-500/dataset/42_svk_id.zip",
    "ftp://smartengines.com/midv-500/dataset/43_tur_id.zip",
    "ftp://smartengines.com/midv-500/dataset/44_ukr_id.zip",
    "ftp://smartengines.com/midv-500/dataset/45_ukr_passport.zip",
    "ftp://smartengines.com/midv-500/dataset/46_ury_passport.zip",
    "ftp://smartengines.com/midv-500/dataset/47_usa_bordercrossing.zip",
    "ftp://smartengines.com/midv-500/dataset/48_usa_passportcard.zip",
    "ftp://smartengines.com/midv-500/dataset/49_usa_ssn82.zip",
    "ftp://smartengines.com/midv-500/dataset/50_xpo_id.zip",
]


midv2019_links: list[str] = [
    "ftp://smartengines.com/midv-500/extra/midv-2019/dataset/01_alb_id.zip",
    "ftp://smartengines.com/midv-500/extra/midv-2019/dataset/02_aut_drvlic_new.zip",
    "ftp://smartengines.com/midv-500/extra/midv-2019/dataset/03_aut_id_old.zip",
    "ftp://smartengines.com/midv-500/extra/midv-2019/dataset/04_aut_id.zip",
    "ftp://smartengines.com/midv-500/extra/midv-2019/dataset/05_aze_passport.zip",
    "ftp://smartengines.com/midv-500/extra/midv-2019/dataset/06_bra_passport.zip",
    "ftp://smartengines.com/midv-500/extra/midv-2019/dataset/07_chl_id.zip",
    "ftp://smartengines.com/midv-500/extra/midv-2019/dataset/08_chn_homereturn.zip",
    "ftp://smartengines.com/midv-500/extra/midv-2019/dataset/09_chn_id.zip",
    "ftp://smartengines.com/midv-500/extra/midv-2019/dataset/10_cze_id.zip",
    "ftp://smartengines.com/midv-500/extra/midv-2019/dataset/11_cze_passport.zip",
    "ftp://smartengines.com/midv-500/extra/midv-2019/dataset/12_deu_drvlic_new.zip",
    "ftp://smartengines.com/midv-500/extra/midv-2019/dataset/13_deu_drvlic_old.zip",
    "ftp://smartengines.com/midv-500/extra/midv-2019/dataset/14_deu_id_new.zip",
    "ftp://smartengines.com/midv-500/extra/midv-2019/dataset/15_deu_id_old.zip",
    "ftp://smartengines.com/midv-500/extra/midv-2019/dataset/16_deu_passport_new.zip",
    "ftp://smartengines.com/midv-500/extra/midv-2019/dataset/17_deu_passport_old.zip",
    "ftp://smartengines.com/midv-500/extra/midv-2019/dataset/18_dza_passport.zip",
    "ftp://smartengines.com/midv-500/extra/midv-2019/dataset/19_esp_drvlic.zip",
    "ftp://smartengines.com/midv-500/extra/midv-2019/dataset/20_esp_id_new.zip",
    "ftp://smartengines.com/midv-500/extra/midv-2019/dataset/21_esp_id_old.zip",
    "ftp://smartengines.com/midv-500/extra/midv-2019/dataset/22_est_id.zip",
    "ftp://smartengines.com/midv-500/extra/midv-2019/dataset/23_fin_drvlic.zip",
    "ftp://smartengines.com/midv-500/extra/midv-2019/dataset/24_fin_id.zip",
    "ftp://smartengines.com/midv-500/extra/midv-2019/dataset/25_grc_passport.zip",
    "ftp://smartengines.com/midv-500/extra/midv-2019/dataset/26_hrv_drvlic.zip",
    "ftp://smartengines.com/midv-500/extra/midv-2019/dataset/27_hrv_passport.zip",
    "ftp://smartengines.com/midv-500/extra/midv-2019/dataset/28_hun_passport.zip",
    "ftp://smartengines.com/midv-500/extra/midv-2019/dataset/29_irn_drvlic.zip",
    "ftp://smartengines.com/midv-500/extra/midv-2019/dataset/30_ita_drvlic.zip",
    "ftp://smartengines.com/midv-500/extra/midv-2019/dataset/31_jpn_drvlic.zip",
    "ftp://smartengines.com/midv-500/extra/midv-2019/dataset/32_lva_passport.zip",
    "ftp://smartengines.com/midv-500/extra/midv-2019/dataset/33_mac_id.zip",
    "ftp://smartengines.com/midv-500/extra/midv-2019/dataset/34_mda_passport.zip",
    "ftp://smartengines.com/midv-500/extra/midv-2019/dataset/35_nor_drvlic.zip",
    "ftp://smartengines.com/midv-500/extra/midv-2019/dataset/36_pol_drvlic.zip",
    "ftp://smartengines.com/midv-500/extra/midv-2019/dataset/37_prt_id.zip",
    "ftp://smartengines.com/midv-500/extra/midv-2019/dataset/38_rou_drvlic.zip",
    "ftp://smartengines.com/midv-500/extra/midv-2019/dataset/39_rus_internalpassport.zip",
    "ftp://smartengines.com/midv-500/extra/midv-2019/dataset/40_srb_id.zip",
    "ftp://smartengines.com/midv-500/extra/midv-2019/dataset/41_srb_passport.zip",
    "ftp://smartengines.com/midv-500/extra/midv-2019/dataset/42_svk_id.zip",
    "ftp://smartengines.com/midv-500/extra/midv-2019/dataset/43_tur_id.zip",
    "ftp://smartengines.com/midv-500/extra/midv-2019/dataset/44_ukr_id.zip",
    "ftp://smartengines.com/midv-500/extra/midv-2019/dataset/45_ukr_passport.zip",
    "ftp://smartengines.com/midv-500/extra/midv-2019/dataset/46_ury_passport.zip",
    "ftp://smartengines.com/midv-500/extra/midv-2019/dataset/47_usa_bordercrossing.zip",
    "ftp://smartengines.com/midv-500/extra/midv-2019/dataset/48_usa_passportcard.zip",
    "ftp://smartengines.com/midv-500/extra/midv-2019/dataset/49_usa_ssn82.zip",
    "ftp://smartengines.com/midv-500/extra/midv-2019/dataset/50_xpo_id.zip",
]


def _iter_zip_links(dataset_name: str) -> dict[str, list[str]]:
    if dataset_name == "midv500":
        return {"midv500": midv500_links}
    if dataset_name == "midv2019":
        return {"midv2019": midv2019_links}
    if dataset_name == "all":
        return {"midv500": midv500_links, "midv2019": midv2019_links}

    raise ValueError('Invalid dataset_name. Use "midv500", "midv2019" or "all".')


def _is_non_empty_dir(path: Path) -> bool:
    if not path.exists() or not path.is_dir():
        return False
    try:
        next(path.iterdir())
        return True
    except StopIteration:
        return False


def _fetch_one(link: str, dst: Path, skip_existing: bool, cleanup_zip: bool) -> None:
    """Download and extract one dataset ZIP into `dst`."""
    print("-" * 70)
    filename = link.split("/")[-1]
    zip_path = dst / filename
    extracted_dir = dst / filename.removesuffix(".zip")

    if skip_existing and _is_non_empty_dir(extracted_dir):
        print(f"Already extracted -> skipping: {extracted_dir}")
        return

    if zip_path.exists():
        print(f"ZIP already present -> skipping download: {zip_path.name}")
    else:
        print(f"Downloading: {filename}")
        download(link, dst)
        print(f"Downloaded: {filename}")

    print(f"Unzipping: {filename}")
    unzip(zip_path, dst)
    print(f"Unzipped: {extracted_dir.name}")

    if cleanup_zip:
        try:
            zip_path.unlink()
            print(f"Removed ZIP: {zip_path.name}")
        except OSError:
            # Non-fatal; leave the ZIP behind if deletion fails.
            pass


def download_dataset(
    download_dir: str | Path,
    dataset_name: str = "midv500",
    skip_existing: bool = True,
    cleanup_zip: bool = True,
) -> None:
    """Download and extract the MIDV-500 dataset.

    Important behavior:
    - If `skip_existing` is True and the dataset folder for a ZIP already exists
      (e.g. `.../midv500/01_alb_id/`) and is non-empty, the ZIP will not be
      downloaded again.

    Args:
        download_dir: Output directory.
        dataset_name: One of "midv500", "midv2019", "all".
        skip_existing: Skip items that are already extracted.
        cleanup_zip: Delete the downloaded ZIP after successful extraction.
    """

    base = Path(download_dir)

    for set_name, links in _iter_zip_links(dataset_name).items():
        dst = base / set_name
        dst.mkdir(parents=True, exist_ok=True)

        for link in links:
            _fetch_one(link, dst, skip_existing, cleanup_zip)


# ── Face detection test ───────────────────────────────────────


@dataclass
class RunStats:
    """Accumulated counters for one dataset run."""

    processed: int = 0
    ok: int = 0
    vis_saved: int = 0
    faces_saved_total: int = 0
    faces_per_image: list[int] = field(default_factory=list)
    failures: list[tuple[Path, str]] = field(default_factory=list)


def _find_set_roots(root: Path) -> list[Path]:
    """Return set roots under *root*.

    Accepts:
    - a parent folder containing ``midv500/`` and/or ``midv2019/``
    - a set folder itself (e.g. ``.../midv500/``)
    - a document folder (contains ``images/`` and ``ground_truth/``)
    """
    if (root / "images").is_dir():
        return [root]

    if any((root / d).is_dir() for d in ("01_alb_id", "02_aut_drvlic_new")):
        return [root]

    sets: list[Path] = []
    for name in ("midv500", "midv2019"):
        p = root / name
        if p.is_dir():
            sets.append(p)

    return sets or [root]


def _find_document_roots(set_root: Path) -> list[Path]:
    if not set_root.is_dir():
        return []

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
    """Extract face crops, save them, return ``(saved_count, faces)``.

    ``faces`` covers the same (possibly truncated) set as the crops
    considered for saving, so callers' counts stay consistent.
    """
    extracted = detector.extract_faces(image_path, padding=face_padding)

    if max_faces_per_image > 0:
        extracted = extracted[:max_faces_per_image]

    faces = [(x, y, w, h, conf) for (_img, (x, y, w, h), conf) in extracted]

    saved = 0
    for idx, (face_img, _bbox, conf) in enumerate(extracted, 1):
        if max_saved_total > 0 and global_saved + saved >= max_saved_total:
            break

        crop_dir = out_dir / doc_name
        crop_dir.mkdir(parents=True, exist_ok=True)

        out_path = crop_dir / (f"{image_path.stem}_face_{idx:02d}_conf_{conf:.3f}.jpg")
        if cv2.imwrite(str(out_path), face_img):
            saved += 1

    return saved, faces


def _find_all_doc_roots(root: Path, max_docs: int) -> tuple[list[Path], list[Path]]:
    """Return (set_roots, doc_roots) under *root*, honoring *max_docs*."""
    set_roots = _find_set_roots(root)
    doc_roots: list[Path] = []
    for sr in set_roots:
        doc_roots.extend(_find_document_roots(sr))
    if max_docs > 0:
        doc_roots = doc_roots[:max_docs]
    return set_roots, doc_roots


def _iter_selected_images(
    doc_roots: list[Path],
    max_images_per_doc: int,
    max_images_total: int,
    include_root_images: bool,
) -> Iterable[tuple[Path, Path]]:
    """Yield (doc_root, image_path) pairs within the configured limits."""
    processed = 0
    for doc_root in doc_roots:
        image_paths = _iter_image_paths(doc_root, include_root_images)
        if max_images_per_doc > 0:
            image_paths = image_paths[:max_images_per_doc]
        print(f"Doc: {doc_root.name} | images_selected={len(image_paths)}")

        for image_path in image_paths:
            if max_images_total > 0 and processed >= max_images_total:
                return
            processed += 1
            yield doc_root, image_path


def _detect_or_save(
    detector: FaceDetectorDNN,
    image_path: Path,
    doc_name: str,
    save_faces: bool,
    faces_dir: Path,
    max_faces_per_image: int,
    face_padding: float,
    faces_saved_total: int,
    max_saved_faces_total: int,
) -> tuple[list[tuple[int, int, int, int, float]], int]:
    """Detect faces, or extract and save crops. Returns (faces, crops_saved)."""
    if not save_faces:
        return detector.detect_face(image_path), 0

    saved, faces = _save_face_crops(
        detector,
        image_path=image_path,
        out_dir=faces_dir,
        doc_name=doc_name,
        max_faces_per_image=max_faces_per_image,
        face_padding=face_padding,
        global_saved=faces_saved_total,
        max_saved_total=max_saved_faces_total,
    )
    return faces, saved


def _process_one_image(
    detector: FaceDetectorDNN,
    doc_root: Path,
    image_path: Path,
    stats: RunStats,
    save_vis: int,
    results_dir: Path | None,
    save_faces: bool,
    faces_dir: Path,
    max_faces_per_image: int,
    face_padding: float,
    max_saved_faces_total: int,
) -> None:
    """Process a single image and update *stats*."""
    stats.processed += 1
    try:
        faces, saved = _detect_or_save(
            detector,
            image_path,
            doc_root.name,
            save_faces,
            faces_dir,
            max_faces_per_image,
            face_padding,
            stats.faces_saved_total,
            max_saved_faces_total,
        )
    except BlitzIDError as e:
        stats.failures.append((image_path, str(e)))
        return
    except Exception as e:
        stats.failures.append((image_path, f"Unexpected error: {e}"))
        return

    stats.ok += 1
    stats.faces_saved_total += saved
    stats.faces_per_image.append(len(faces))

    if results_dir is not None and stats.vis_saved < save_vis and faces:
        tag = f"{doc_root.name}_{image_path.stem}_faces_{len(faces)}"
        out_path = _save_visualization(detector, image_path, faces, results_dir, tag)
        stats.vis_saved += 1
        print(f"  saved_vis: {out_path}")


def _print_summary(
    set_roots: list[Path],
    doc_roots: list[Path],
    stats: RunStats,
    save_faces: bool,
    faces_dir: Path,
) -> None:
    """Print run counters and the first failures."""
    print("\nSummary:")
    print(f"  set_roots:           {len(set_roots)}")
    print(f"  doc_roots:           {len(doc_roots)}")
    print(f"  images_processed:    {stats.processed}")
    print(f"  ok:                  {stats.ok}")
    print(f"  failed:              {len(stats.failures)}")
    print(f"  faces_total:         {sum(stats.faces_per_image)}")
    print(f"  vis_saved:           {stats.vis_saved}")
    if save_faces:
        print(f"  faces_saved_total:   {stats.faces_saved_total} -> {faces_dir}")

    if stats.faces_per_image:
        print("  faces/image stats:")
        print(f"    mean:   {statistics.mean(stats.faces_per_image):.3f}")
        print(f"    median: {statistics.median(stats.faces_per_image):.3f}")
        print(f"    max:    {max(stats.faces_per_image)}")
        print(f"    min:    {min(stats.faces_per_image)}")

    if stats.failures:
        print("\nFirst failures:")
        for p, msg in stats.failures[:10]:
            print(f"- {p}: {msg}")


def run_face_test(
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
    log_level: int,
) -> int:
    try:
        detector = FaceDetectorDNN(log_level=log_level)
    except Exception as e:
        print(f"Failed to initialise FaceDetectorDNN: {e}")
        return 2

    set_roots, doc_roots = _find_all_doc_roots(root, max_docs)
    if not doc_roots:
        print(
            f"No MIDV-500 document roots found under: {root}\n"
            "Expected folders containing 'images/' and 'ground_truth/'."
        )
        return 2

    stats = RunStats()
    for doc_root, image_path in _iter_selected_images(
        doc_roots, max_images_per_doc, max_images_total, include_root_images
    ):
        _process_one_image(
            detector,
            doc_root,
            image_path,
            stats,
            save_vis,
            results_dir,
            save_faces,
            faces_dir,
            max_faces_per_image,
            face_padding,
            max_saved_faces_total,
        )

    _print_summary(set_roots, doc_roots, stats, save_faces, faces_dir)
    return 0 if not stats.failures else 1


# ── CLI ──────────────────────────────────────────────────────


def _parse_log_level(value: str) -> int:
    import logging

    mapping = {
        "CRITICAL": logging.CRITICAL,
        "ERROR": logging.ERROR,
        "WARNING": logging.WARNING,
        "INFO": logging.INFO,
        "DEBUG": logging.DEBUG,
    }
    key = value.strip().upper()
    if key not in mapping:
        raise argparse.ArgumentTypeError(
            f"Invalid log level: {value}. Choose from: {', '.join(mapping)}"
        )
    return mapping[key]


def _build_download_parser(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "download_dir",
        nargs="?",
        default="data",
        help='Directory where the dataset will be stored (default: "data").',
    )
    parser.add_argument(
        "--dataset",
        dest="dataset_name",
        choices=["midv500", "midv2019", "all"],
        default="midv500",
        help="Which dataset set to download.",
    )
    parser.add_argument(
        "--no-skip-existing",
        dest="skip_existing",
        action="store_false",
        help="Do not skip already-extracted folders; re-download/re-extract.",
    )
    parser.add_argument(
        "--keep-zip",
        dest="cleanup_zip",
        action="store_false",
        help="Keep downloaded ZIP files after extraction.",
    )
    parser.set_defaults(skip_existing=True, cleanup_zip=True)


def _build_test_parser(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "root",
        nargs="?",
        default="data/midv500",
        help="Dataset root (parent with midv500/, a set root, or a document root).",
    )
    parser.add_argument("--max-docs", type=int, default=3)
    parser.add_argument("--max-images-per-doc", type=int, default=20)
    parser.add_argument("--max-images-total", type=int, default=100)

    parser.add_argument(
        "--results-dir", type=Path, default=Path("results/midv500_face_test")
    )
    parser.add_argument("--save-vis", type=int, default=5)

    parser.add_argument("--save-faces", action="store_true")
    parser.add_argument(
        "--faces-dir", type=Path, default=Path("results/midv500_face_crops")
    )
    parser.add_argument("--max-saved-faces-total", type=int, default=200)
    parser.add_argument("--max-faces-per-image", type=int, default=5)
    parser.add_argument("--face-padding", type=float, default=0.2)

    parser.add_argument("--include-root-images", action="store_true")
    parser.add_argument("--log-level", default="WARNING")


def main(argv: Iterable[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="MIDV-500 dataset tooling: download and face-detection test."
    )
    sub = parser.add_subparsers(dest="command", required=True)

    _build_download_parser(sub.add_parser("download", help="Download and extract."))
    _build_test_parser(
        sub.add_parser("test", help="Run FaceDetectorDNN over the dataset.")
    )

    args = parser.parse_args(list(argv) if argv is not None else None)

    if args.command == "download":
        download_dataset(
            download_dir=args.download_dir,
            dataset_name=args.dataset_name,
            skip_existing=args.skip_existing,
            cleanup_zip=args.cleanup_zip,
        )
        return

    results_dir = None if args.save_vis <= 0 else Path(args.results_dir)
    code = run_face_test(
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
        log_level=_parse_log_level(str(args.log_level)),
    )
    raise SystemExit(code)


if __name__ == "__main__":
    main()
