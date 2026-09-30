# Scripts

Demo, diagnostic, and benchmark scripts. Not part of the shipped
package. All commands run from the repo root with the project
environment: `uv sync --all-extras` first (scripts need the `scripts`
extra; OCR demos need the `ocr` extra).

Model weights (SCRFD, PP-OCR) download on first use into the
platformdirs cache — never into the repo.

## `face_detector_demo.py` — face detection demo

Single-image or batch runs over `images/`, using the shared CLI in
`demo_cli.py` (`--preset`, `--image`, `--batch-dir`, `--run`,
`--confidence-threshold`, `--min-face-size`, `--padding`, `--cache`,
`--max-cache-size`, `--model-dir`, `--results-dir`, `--max-images`,
`--log-level`).

```bash
# single image, all runs, balanced preset
uv run python scripts/face_detector_demo.py --preset balanced --run all \
    --image images/bub_der_personalausweis_kopie.jpg

# batch over the committed fixtures
uv run python scripts/face_detector_demo.py
```

## `ocr_demo.py` — OCR text reading demo (needs `ocr` extra)

Prints every recognized text line with bbox and confidence.

```bash
uv run python scripts/ocr_demo.py --image images/nl_td1_id_specimen.jpg
uv run python scripts/ocr_demo.py --image images/td3_passport_specimen.jpg
```

## `midv500_download.py` — MIDV-500 dataset download (needs `scripts` extra)

Downloads the MIDV-500 / MIDV-2019 dataset into a local data directory.
Datasets are runtime artifacts: keep them out of the repo.

```bash
uv run python scripts/midv500_download.py --dataset midv500 data/
```

## `midv500_face_test.py` — face detection over MIDV-500 (needs `scripts` extra)

Runs `FaceDetectorDNN` across a downloaded MIDV-500 tree, reporting
detection statistics; optionally saves crops and visualizations
(`--save-faces`, `--save-vis`, `--faces-dir`, `--results-dir`, plus
`--max-docs` / `--max-images-per-doc` / `--max-images-total` limits to
bound runtime).

```bash
uv run python scripts/midv500_face_test.py data/midv500 --max-docs 3
```

## PII

These scripts process document imagery. Keep real photos and dataset
samples in untracked local directories (`data/` is gitignored) — never
commit them. The committed `images/` fixtures are specimen documents
and public-domain photos, attributed in `images/README.md`.
