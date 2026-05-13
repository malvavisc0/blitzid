from __future__ import annotations

import argparse
import json
import os
import zipfile
from collections.abc import Iterable
from pathlib import Path
from urllib.request import urlretrieve

from tqdm import tqdm


def calculate_intersect_area(bbox1: list[float], bbox2: list[float]) -> float:
    """Return the intersection area of two axis-aligned bounding boxes.

    Each bbox is `[xmin, ymin, xmax, ymax]`.
    """

    x_a = max(bbox1[0], bbox2[0])
    y_a = max(bbox1[1], bbox2[1])
    x_b = min(bbox1[2], bbox2[2])
    y_b = min(bbox1[3], bbox2[3])

    inter_w = max(x_b - x_a, 0.0)
    inter_h = max(y_b - y_a, 0.0)
    return inter_w * inter_h


def get_bbox_inside_image(
    label_bbox: list[float], image_bbox: list[float]
) -> list[float]:
    """Clamp `label_bbox` so it lies inside `image_bbox`.

    Returns `[xmin, ymin, xmax, ymax]`.
    """

    x_a = max(label_bbox[0], image_bbox[0])
    y_a = max(label_bbox[1], image_bbox[1])
    x_b = min(label_bbox[2], image_bbox[2])
    y_b = min(label_bbox[3], image_bbox[3])
    return [x_a, y_a, x_b, y_b]


def list_annotation_paths_recursively(
    directory: str,
    ignore_background_only_ones: bool = True,
    image_size: tuple[int, int] = (1080, 1920),
) -> list[str]:
    """List per-frame annotation JSON files under `directory`.

    Notes:
    - MIDV-500 stores annotations as JSON ("quad" polygons) under `ground_truth/`.
    - If `ignore_background_only_ones` is True, annotations that do not intersect
      the image at all are discarded.

    Returns:
        A list of paths relative to `directory`.
    """

    image_w, image_h = image_size
    image_bbox = [0.0, 0.0, float(image_w), float(image_h)]

    relative_filepath_list: list[str] = []

    for root, _, files in os.walk(directory):
        for file in files:
            if not file.endswith(".json"):
                continue

            abs_filepath = os.path.join(root, file)

            # Skip sample id-card json like "43_tur_id.json" (dataset-level metadata).
            if "id" in os.path.basename(abs_filepath):
                continue

            try:
                with open(abs_filepath, encoding="utf-8") as json_file:
                    quad = json.load(json_file)
                coords = quad["quad"]
            except (OSError, json.JSONDecodeError, KeyError, TypeError):
                # Known oddities exist in the dataset (e.g. some malformed JSON files).
                continue

            label_xmin = min(pos[0] for pos in coords)
            label_xmax = max(pos[0] for pos in coords)
            label_ymin = min(pos[1] for pos in coords)
            label_ymax = max(pos[1] for pos in coords)

            label_bbox = [label_xmin, label_ymin, label_xmax, label_ymax]
            if ignore_background_only_ones:
                intersect_area = calculate_intersect_area(label_bbox, image_bbox)
                if intersect_area < 1.0:
                    continue

            abs_filepath = abs_filepath.replace("\\", "/")  # for Windows
            relative_filepath = abs_filepath.split(directory)[-1]
            if relative_filepath.startswith("/"):
                relative_filepath = relative_filepath[1:]
            relative_filepath_list.append(relative_filepath)

    number_of_files = len(relative_filepath_list)
    folder_name = Path(directory).name
    print(f"There are {number_of_files} annotation json files in folder {folder_name}.")

    return relative_filepath_list


def create_dir(dir_path: str | Path) -> None:
    """Create `dir_path` if it doesn't exist."""

    Path(dir_path).mkdir(parents=True, exist_ok=True)


class TqdmUpTo(tqdm):
    """`tqdm` progress bar compatible with `urllib.request.urlretrieve()` hooks."""

    def update_to(self, b: int = 1, bsize: int = 1, tsize: int | None = None) -> None:
        """Update the progress bar with `b` blocks of size `bsize`."""
        if tsize is not None:
            self.total = tsize
        self.update(b * bsize - self.n)


def download(url: str, save_dir: str | Path) -> Path:
    """Download a file to `save_dir` and return the local path."""

    save_dir = Path(save_dir)
    create_dir(save_dir)

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

    # Inline branching (instead of `_iter_zip_links()`) keeps older Pylint versions
    # happy about definite assignment.
    if dataset_name == "midv500":
        links_set: dict[str, list[str]] = {"midv500": midv500_links}
    elif dataset_name == "midv2019":
        links_set = {"midv2019": midv2019_links}
    elif dataset_name == "all":
        links_set = {"midv500": midv500_links, "midv2019": midv2019_links}
    else:
        raise ValueError('Invalid dataset_name. Use "midv500", "midv2019" or "all".')

    base = Path(download_dir)

    for set_name, links in links_set.items():
        dst = base / set_name
        dst.mkdir(parents=True, exist_ok=True)

        for link in links:
            print("-" * 70)
            link = link.replace("\\", "/")  # for Windows
            filename = link.split("/")[-1]
            zip_path = dst / filename
            extracted_dir = dst / filename.removesuffix(".zip")

            if skip_existing and _is_non_empty_dir(extracted_dir):
                print(f"Already extracted -> skipping: {extracted_dir}")
                continue

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


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Download and extract the MIDV-500 dataset (Smart Engines)."
    )

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
    return parser


def main(argv: Iterable[str] | None = None) -> None:
    """Download and extract the MIDV-500 or MIDV-2019 dataset."""
    parser = _build_arg_parser()
    args = parser.parse_args(list(argv) if argv is not None else None)

    download_dataset(
        download_dir=args.download_dir,
        dataset_name=args.dataset_name,
        skip_existing=args.skip_existing,
        cleanup_zip=args.cleanup_zip,
    )


if __name__ == "__main__":
    main()
