# Scripts

Demo, diagnostic, and benchmark scripts. Not part of the shipped
package. All commands run from the repo root with the project
environment: `uv sync --all-extras` first (scripts need the `scripts`
extra; OCR demos need the `ocr` extra).

Model weights (SCRFD, PP-OCR) download on first use into the
platformdirs cache — never into the repo.

All output-producing scripts write into a results directory (default
`results/`, gitignored local artifact) in addition to printing:
face crops and visualizations, OCR lines, benchmark tables.

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

Prints every recognized text line with bbox and confidence, and writes
them to `results/ocr-<image>-<timestamp>.md`.

```bash
uv run python scripts/ocr_demo.py --image images/nl_td1_id_specimen.jpg
uv run python scripts/ocr_demo.py --image images/td3_passport_specimen.jpg
```

## `structurize_smoke.py` — structured-extraction smoke test (needs `ocr` extra)

Calls the configured LLM endpoint (default a local vLLM server) on
three document kinds — sample MRZ lines, a specimen ID image, and a
specimen license-plate image — asserts the `document_type`
classification (`mrz` / `id` / `plate`), and prints the extracted
fields with confidences. This is the only place that makes endpoint
calls — the pytest suite never hits the network. Configure the endpoint
via `BLITZID_LLM_BASE_URL` / `BLITZID_LLM_API_KEY` / `BLITZID_LLM_MODEL`.

```bash
uv run python scripts/structurize_smoke.py
```

## `benchmark.py` — pipeline timing

Times engine init and warm per-call latency for the detection presets,
the result cache, and (with the `ocr` extra) OCR/MRZ reading. Prints a
table and writes it to `results/benchmark-<timestamp>.md`.

```bash
uv run python scripts/benchmark.py --markdown
```

## `midv500.py` — MIDV-500 dataset tooling (needs `scripts` extra)

One script, two subcommands. `download` fetches the MIDV-500 /
MIDV-2019 dataset into a local data directory; `test` runs
`FaceDetectorDNN` across a downloaded tree, reporting detection
statistics; optionally saves crops and visualizations
(`--save-faces`, `--save-vis`, `--faces-dir`, `--results-dir`, plus
`--max-docs` / `--max-images-per-doc` / `--max-images-total` limits to
bound runtime). Datasets are runtime artifacts: keep them out of the
repo.

```bash
uv run python scripts/midv500.py download data/
uv run python scripts/midv500.py test data/midv500 --max-docs 3
```

## PII

These scripts process document imagery. Keep real photos and dataset
samples in untracked local directories (`data/` is gitignored) — never
commit them. The committed `images/` fixtures are specimen documents
and public-domain photos, attributed in `images/README.md`.
