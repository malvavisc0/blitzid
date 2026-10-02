# Changelog

All notable changes to this project will be documented in this file.

## [Unreleased]

## [0.1.1] - 2026-10-01

### Added

- 1:1 face verification: `FaceVerifier` aligns detected faces to the
  canonical ArcFace template via their five SCRFD landmarks, embeds
  them with the ArcFace recognizer (InsightFace `buffalo_m` recognition
  weights, ~174 MB, 512-d, downloaded on first use into the models dir
  — `BLITZID_MODELS_DIR` when set), and compares the L2-normalized
  vectors by cosine similarity against a decision threshold (default
  `0.4`). API: `verify(image1, image2)` (best face per image),
  `verify_faces(image1, face1, image2, face2)` (reuses detections),
  `embed(image, face=None)`, and static `similarity(e1, e2)`; exports
  `FaceVerifier`, `VerificationResult`, and `FaceVerificationError`
  from the package root. The ArcFace weights load from `model_dir`,
  defaulting to a supplied detector's models directory, else the
  default models dir (`FaceVerifier.model_manager` mirrors the
  detector's). `ModelManager` gained `filename`/`url`/
  `sha256` constructor overrides (SCRFD defaults unchanged) and now
  verifies downloads against a pinned digest (SCRFD and ArcFace both);
  `FaceDetectorDNN` / `FaceVerifier` expose `allow_downloads`;
  `scripts/download_models.py` pre-fetches the recognition weights too.
  Smoke test: `scripts/verify_smoke.py` (specimen ID portrait, rescaled
  copy, and two distinct faces from the conference fixture).
- Face attributes (`age`, `gender`, `race` per face):
  `FaceAttributeReader` / `FaceAttributes` (FairFace weights with a
  pinned digest) and an `attributes` analysis type in the HTTP API.
- Document self-consistency: the `consistency` analysis type
  cross-checks the printed fields against the machine-readable zone
  (`cross_check`, `ConsistencyReport`, `FieldComparison`) and the
  portrait's age band and gender against the birth date and sex
  (`PhotoComparison`), with per-field verdicts and both values.
- LLM-backed structured extraction (`ocr` extra): `StructuredOCRReader`
  rebuilds a typed record from OCR text lines via an OpenAI-compatible
  chat endpoint (pydantic-ai; endpoint via `BLITZID_LLM_BASE_URL` /
  `BLITZID_LLM_API_KEY` / `BLITZID_LLM_MODEL` or the matching
  constructor arguments — all required, no baked-in defaults). Exports `StructuredOCRReader`, `StructuredOCR`,
  and `ExtractedField` from the package root; `pydantic` becomes a core
  dependency for the exported schemas. `document_type` is a closed
  vocabulary (`id`, `plate`, `mrz`, `unknown`); date-named fields come
  back as `datetime.date`; the reader stamps `raw_text` from the actual
  prompt, upper-cases text values itself, and collapses duplicate field
  names to the highest-confidence entry. The endpoint call is bounded
  by `timeout` (default 180 s). `read(texts, kind=...)` selects a
  specialized per-kind prompt (`id` / `plate` / `mrz`, stamping
  `document_type`) or `kind="auto"` for model classification. Optional
  Langfuse tracing of the LLM calls (`langfuse` in the `ocr` extra),
  opt-in via the standard `LANGFUSE_PUBLIC_KEY` /
  `LANGFUSE_SECRET_KEY` / `LANGFUSE_BASE_URL` environment variables.
  Smoke test: `scripts/structurize_smoke.py` (MRZ lines, specimen ID
  image, specimen license-plate image).

### Changed

- The MRZ reader resolves obvious OCR confusions (0/O, 1/I, 2/Z,
  5/S, 6/G, 8/B) per field alphabet before check-digit validation
  instead of rejecting them; unresolvable input still raises
  `MRZError`. Structured extraction builds its agent per call (a
  reused agent's HTTP client broke on later calls from other threads),
  and one job's sections share detection, OCR, and LLM work.
- `Face.landmarks` now carry subpixel float coordinates (previously
  truncated to integers), so ArcFace alignment uses the detector's full
  keypoint precision. `FaceDetectorDNN.detect_from_array` is public
  (was the private `_detect_from_array`); `ModelError` messages for a
  missing model now name the full path looked up.

## [0.1.0] - 2026-10-01

### Added

- HTTP API (`blitzid[api]` extra: fastapi, uvicorn, python-multipart,
  redis) exposing the analyses as async jobs: `POST /analyze`
  (multipart upload + `types` form field; eager validation with
  422/400/413/415/503), `GET /jobs/{id}` (claim-once results via
  GETDEL — later reads get `410 Gone`; TTL-bounded), `GET /health`
  for orchestration (`503` when the job store is unreachable). Jobs
  run in per-process worker threads bounded by
  `BLITZID_API_MAX_CONCURRENT_JOBS` with per-engine locks; a Redis
  queue (persistence off — RAM only) with an atomically enforced cap,
  TTL, raw-byte job payloads, and token-owned claims: worker
  heartbeats extend the claim lease, a lost claim discards that
  worker's result (the re-claimed worker's result wins), and the
  claim-lease sweeper re-runs only jobs whose worker actually died. Env
  knobs: `BLITZID_API_REDIS_URL`, `BLITZID_API_MAX_UPLOAD_MB`,
  `BLITZID_API_JOB_TTL_SECONDS`, `BLITZID_API_MAX_QUEUED_JOBS`,
  `BLITZID_API_MAX_CONCURRENT_JOBS`, `BLITZID_API_JOB_LEASE_SECONDS`.
- Docker deployment: multi-stage `Dockerfile` (weights baked to
  `/models`, non-root, python-urllib HEALTHCHECK) and
  `docker-compose.yml` (api + RAM-only `redis:8-alpine` with
  persistence disabled). Tag pushes publish
  `ghcr.io/malvavisc0/blitzid` via `.github/workflows/docker.yml`,
  gated on the (now reusable) CI gate plus a compose smoke test that
  submits a real face/ocr/mrz job and polls it to completion.
- PyPI publishing: tag pushes matching the package version build and
  upload the sdist/wheel to PyPI via `.github/workflows/publish.yml`
  (trusted publishing — no API tokens), gated on the same CI matrix.
- `DocumentCropper` / `QualityReport` (reading domain, pure CV):
  locates the largest plausible document quad via Canny edges +
  contour approximation, warps it into an axis-aligned canonical crop
  at the quad's own aspect, and quality-checks it (`document_found`,
  `aspect_ratio` near ID-1/ID-3, `resolution`, `sharpness`,
  `brightness`, side-aware `face_present` — never fails a document).
  Optional `FaceDetectorDNN` injection powers the face check
  (`front` + no face → warn; `back` → not expected) without adding a
  runtime `reading → face` dependency; an optional `detector_lock`
  serializes the check against other users of a shared detector. The
  `POST /crop` API endpoint wraps it: QC verdict plus the canonical
  `crop_base64` JPEG.
- `BLITZID_MODELS_DIR` environment variable: relocates all model-weight
  downloads (SCRFD + RapidOCR) when set; defaults to the platformdirs
  user cache as before. `scripts/download_models.py` pre-fetches all
  weights into a directory (default `models/`) for Docker images.
- `MRZReader` / `MRZRecord` / `MRZError`: ICAO 9303 machine-readable zone
  reading on top of the `ocr` extra. Selects MRZ lines from OCR output
  (30/36/44 chars over the MRZ charset, reassembled by bbox position),
  validates every check digit plus the letter-only fields (document
  code, issuer, nationality — the ones without check-digit protection),
  and parses TD1, TD2, and TD3 layouts. No fuzzy OCR-error correction:
  an unreadable zone raises `MRZError` with the failing field.
  Document numbers longer than their field follow the ICAO
  convention — no printed check digit, overflow at the start of
  `optional_data1` — and are accepted (the composite check digit
  still covers the full number).
- `RapidOCRReader` / `OCRText` and the `blitzid[ocr]` extra: OCR text
  reading for ID documents via RapidOCR (PP-OCR ONNX models on
  onnxruntime CPU). Models download on first use to
  `user_cache_dir("blitzid")/models/rapidocr`; `read()` returns text
  lines sorted by confidence. Core installs stay OCR-free.
  `scripts/ocr_demo.py` demonstrates the reader from the CLI.
- `detect_face_landmarks()`, returning the new frozen `Face` dataclass
  (`bbox`, `confidence`, `landmarks`). SCRFD decodes its 5-point keypoint
  head — right eye, left eye, nose tip, right mouth corner, left mouth
  corner — previously discarded. Existing `detect_face()` tuples are
  unchanged.
- Positive-path tests on repo image fixtures: the specimen ID card (exactly
  one face — its portrait) and the 1927 conference photo (29 people). Also
  fails fast with `ModelError` on an unexpected SCRFD ONNX output layout.
- Public `FaceDetectorDNN.cache_enabled` property (scripts no longer read
  the private `_cache` attribute).
- `face_detector_demo.py` validates `--run` demo names and exits with the
  valid set on typos; cache CLI knobs are documented as custom-preset-only.
- Detection caching with configurable LRU cache.
- Multi-scale detection for improved recall on small faces.
- Typed package (`py.typed` marker).

### Changed

- Replaced the SSD ResNet-10 Caffe detector with **SCRFD-2.5G** (InsightFace
  `buffalo_m` ONNX weights) running on **onnxruntime CPU**. OpenCV 5 removed
  the Caffe importer; inference now fails fast with `ModelError` instead of
  falling back to other models or providers.
- `FaceDetectorDNN` gains a `det_size` parameter (default `640x640`, multiples
  of 32) and letterboxes input images so detections map back to original image
  coordinates without aspect distortion.
- `DetectionMetrics.backend` now reports `"ONNXRuntime"`.
- `detect_faces_batch()` now omits failed images from the result dict; an
  empty list always means "processed, zero faces". Failed images are still
  logged. Previously failures were indistinguishable from legitimate
  zero-face results.
- `DetectionMetrics.faces` now carries `Face` records (landmarks
  accessible) instead of legacy `(x, y, w, h, confidence)` tuples.
  **Breaking** for consumers unpacking metrics entries as tuples.
- Cache hits report real elapsed cache-lookup time in
  `DetectionMetrics.processing_time` instead of a hardcoded `0.0`.
- `scales` runs exactly as configured: `_normalize_scales` no longer
  silently injects a `1.0` pass, so `scales=(1.5,)` costs one inference,
  not two.
- Model downloads stream to a `.tmp` file in 64 KB chunks instead of
  reading the whole file into memory; the socket timeout now bounds the
  transfer.

### Fixed

- Model download no longer raises `ModelError` after a successful transfer:
  the post-download size log stat'ed the already-renamed `.tmp` file. Only
  affected fresh installs (cached models never re-downloaded).
- Detection cache keys now cover image `shape` and `dtype` in addition to
  pixel bytes — arrays sharing byte content but differing in layout no
  longer collide on one cached result.
- Restored the 10000px `MAX_DIMENSION` guard in image validation, lost in
  the SCRFD migration; oversized inputs fail fast with `ImageError`
  instead of allocating multi-GB arrays.
- `midv500_face_test.py` face counts now match the crops actually
  considered for saving when `--max-faces-per-image` truncates the list.

### Removed

- The DeepFace backend entirely: `FaceDetectorDeepFace`,
  `blitzid[deepface]` extra, the `deepface`/`tf-keras` dependencies,
  `tests/test_deepface.py`, and the `analyze()` API. SCRFD via onnxruntime
  is the single detection backend.
- CUDA backend support: `require_cuda` parameters, the CUDA/CPU auto-fallback
  chain in `ModelManager.load_network`, the `CUDAConfigError` alias, and
  `scripts/cuda_diagnostics.py`.
- Dead dataset-annotation helpers in `scripts/midv500_download.py` with zero
  call sites (`list_annotation_paths_recursively`, `calculate_intersect_area`,
  `get_bbox_inside_image`).
- `FaceDetectorDNN.backend_type` compat alias (superseded by `backend`).
