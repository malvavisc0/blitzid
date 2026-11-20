# Models

Local, gitignored storage for downloaded model weights — the repo commits
only this README and a `.gitkeep`.

Populate with:

```bash
python scripts/download_models.py            # -> ./models/
python scripts/download_models.py --models-dir /models
```

## Layout

- `face_detection_yunet_2026may.onnx` — YuNet face detector (OpenCV
  Zoo, MIT, ~230 KB); managed by
  [`ModelManager`](../src/blitzid/_models.py).
- `sface_2021dec.onnx` — SFace face recognizer (OpenCV Zoo, Apache 2.0,
  ~39 MB, 128-d embeddings); managed by `ModelManager` with the
  `filename`/`url`/`sha256` overrides.
- `fairface.onnx` — FairFace attribute model (age group, gender, and
  race per face, ~85 MB); managed by `ModelManager` with the
  `filename`/`url`/`sha256` overrides.
- `minifasnet_v2.onnx` — MiniFASNetV2 anti-spoof model (live vs
  printed photo vs screen photo, ~1.7 MB); managed by `ModelManager`
  with the `filename`/`url`/`sha256` overrides.
- `rapidocr/` — PP-OCR text detection, direction classification, and text
  recognition models (core dependency); managed by RapidOCR's downloader,
  redirected from its in-package default.

The YuNet, SFace, FairFace, and MiniFASNet downloads are verified
against pinned SHA-256 digests before being moved into place; RapidOCR
uses its own downloader. Model weights are not covered by blitzid's MIT
license; see [THIRD_PARTY_NOTICES.md](../THIRD_PARTY_NOTICES.md).

## Default location

When unset, weights download on first use to the `platformdirs` user cache
(`~/.cache/blitzid/models/` on Linux). Set `BLITZID_MODELS_DIR` to
relocate all models (e.g. a Docker volume), or pass `model_dir=` to the
detector, verifier, attribute/anti-spoof readers, or `RapidOCRReader`
for per-instance control.

## Docker

Either bake the weights into the image by running the download script
during `docker build`, or mount a volume at `/models` and set
`BLITZID_MODELS_DIR=/models` so containers share one copy instead of
re-downloading on every start.
