# Changelog

All notable changes to this project will be documented in this file.

## [Unreleased]

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

## [0.1.0] - 2025-05-13

### Added

- OpenCV DNN-based face detection with optional CUDA acceleration.
- Optional DeepFace backend (RetinaFace, MTCNN) with alignment.
- Detection caching with configurable LRU cache.
- Multi-scale detection for improved recall on small faces.
- Adaptive luminance preprocessing for low-contrast images.
- `FaceDetectorDNN` and `FaceDetectorDeepFace` public API.
- Typed package (`py.typed` marker).
