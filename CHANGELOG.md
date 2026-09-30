# Changelog

All notable changes to this project will be documented in this file.

## [Unreleased]

### Added

- `MRZReader` / `MRZRecord` / `MRZError`: ICAO 9303 machine-readable zone
  reading on top of the `ocr` extra. Selects MRZ lines from OCR output
  (30/36/44 chars over the MRZ charset, reassembled by bbox position),
  validates every check digit plus the letter-only fields (document
  code, issuer, nationality — the ones without check-digit protection),
  and parses TD1, TD2, and TD3 layouts. No fuzzy OCR-error correction:
  an unreadable zone raises `MRZError` with the failing field.
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
