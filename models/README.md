# Models

Local, gitignored storage for downloaded model weights — the repo commits
only this README and a `.gitkeep`.

Populate with:

```bash
python scripts/download_models.py            # -> ./models/
python scripts/download_models.py --models-dir /models
```

## Layout

- `scrfd_2.5g.onnx` — SCRFD-2.5G face detector (InsightFace `buffalo_m`
  detection weights); managed by
  [`ModelManager`](../src/blitzid/_models.py).
- `arcface_buffalo_m.onnx` — ArcFace face recognizer (InsightFace
  `buffalo_m` recognition weights, ~174 MB, 512-d embeddings); managed
  by `ModelManager` with the `filename`/`url`/`sha256` overrides.
- `rapidocr/` — PP-OCR text detection, direction classification, and text
  recognition models (the `ocr` extra); managed by RapidOCR's downloader,
  redirected from its in-package default.

Downloads are verified against pinned SHA-256 digests before being
moved into place.

## Default location

When unset, weights download on first use to the `platformdirs` user cache
(`~/.cache/blitzid/models/` on Linux). Set `BLITZID_MODELS_DIR` to
relocate all models (e.g. a Docker volume), or pass `model_dir=` to
`FaceDetectorDNN` / `RapidOCRReader` for per-instance control.

## Docker

Either bake the weights into the image by running the download script
during `docker build`, or mount a volume at `/models` and set
`BLITZID_MODELS_DIR=/models` so containers share one copy instead of
re-downloading on every start.
