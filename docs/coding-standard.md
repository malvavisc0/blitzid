# Python Developer AI Agent

## Communication
You are talking to seniors. Be terse — no preamble, no recap, no
hand-holding, no "here's what I did" summaries. State results, not
intentions. Skip explanations unless explicitly asked. Never narrate
tool use. Answer the question; move on.

## Project
blitzid — modular DNN-based ID document reading framework, optimized
for CPU. A typed Python library (`blitzid`, src layout) with two
layers. Face detection (`src/blitzid/face/`): SCRFD-2.5G ONNX model
(InsightFace `buffalo_m` detection weights, auto-downloaded) run via
an onnxruntime CPU session, with LRU result cache, multi-scale
detection, NMS + size filtering, batch processing, and bbox
visualization. Document reading (`src/blitzid/reading/`): RapidOCR
text lines (`ocr` extra), ICAO 9303 MRZ parsing (TD1/TD2/TD3), and
AAMVA PDF417 driver's-license parsing (`barcode` extra).
Inputs are file paths, NumPy arrays, or PIL Images; outputs are
`(x, y, w, h, confidence)` tuples and crops.

## Stack
- python 3.12+ (CI matrix: 3.12 / 3.13 / 3.14), uv, pyproject.toml
  (hatchling backend, src layout, `py.typed`)
- core deps: numpy, onnxruntime (CPU), opencv-python-headless>=5,
  platformdirs, pydantic
- optional extras: `ocr` (rapidocr, pydantic-ai, langfuse), `barcode`
  (zxing-cpp), `api` (fastapi, uvicorn, redis), `scripts` (tqdm),
  `dev` (pytest, ruff, mypy, pre-commit)
- Type checker: **mypy** (`strict = true` in pyproject.toml)
- Lint/format: **ruff** (line-length 88, target py312; E, W, F, I, UP,
  B, SIM, RUF)
- Tests: **pytest** (`tests/`)
- Hooks: **pre-commit** (ruff --fix, ruff-format, whitespace/yaml/
  large-file checks)

## Commands
```bash
uv sync --extra dev                      # install deps
uv run ruff check --fix src/ tests/ scripts/   # lint
uv run ruff format src/ tests/ scripts/  # format
uv run mypy src/ scripts/                # type check (strict)
uv run pytest tests/ -v                  # test suite
uv run pre-commit run --all-files        # hooks
```
Run `ruff check`, `ruff format --check`, `mypy`, and `pytest` after
every change. CI runs the same on 3.12–3.14.

## Complexity budget
Every function/method stays at cyclomatic complexity rank A or B
(complexity ≤ 10). This applies to the whole codebase, not just new
code: **a pre-existing C/D/E/F block is still a defect and must always
be fixed when encountered** — never worked around, never left "for
later". When a change pushes a block to C+ (or you touch a file
containing one), split it into smaller functions (early returns,
extract helpers, dispatch tables) instead of adding `if` branches, and
fix any pre-existing C+ blocks in the same pass. Verify with:
```bash
uvx radon cc src -s | grep -E '\-\s(C|D|E|F)\s'   # must be empty
```

## Dead code
No dead code — never-read fields, zero-call-site functions, unused
imports, stale aliases. Delete it in the same pass you notice it. A
`_`-prefixed helper with no callers goes out too. The only registered
names are `@pytest.fixture` methods (pytest fixture registry) and mock
attributes like `return_value` (unittest.mock API); everything else
must have a findable call site. Optional dependency shims
(`# pragma: no cover` import fallbacks) are the only code exemption.

## PII / biometrics
This is a face detection library — faces, IDs, and images of real
people are biometric PII. Never use or commit PII — not in code, tests,
docs, plans, examples, or commit messages. Rules:

- Tests, docstrings, and examples use synthetic imagery (`np.zeros(...)`
  arrays, drawn shapes, invented filenames) or the committed fixtures in
  `images/`: official specimen documents and historical public-domain
  photographs of long-deceased persons, both attributed in
  `images/README.md`. Living people's photos stay out — a real person's
  photo in a test fixture is still that person's data.
- Real photos, ID scans, and captured frames stay in local, untracked
  artifacts — never `images/`, never the repo. `images/` never holds
  scans of real documents or photos of living people.
- Datasets (e.g. MIDV-500) are fetched at runtime into cache dirs by
  `scripts/`; never commit downloaded samples or model weights
  (`models/` weights are gitignored — only README/.gitkeep are tracked).
- Before committing anything touched by real imagery, check the diff:
  `git diff --stat` and open every added binary. If it shows a living
  person's face or a real document, it does not go in the repo.

## Priority
Clean, simple, maintainable code. Nothing else.

## Commits
One-line messages only. No body, no trailers.

## Rules
- DRY, KISS
- No unnecessary complexity
- No defensive fallbacks / over-engineering
- Explicit > implicit
- Type hints always (mypy strict must pass)
- PEP 8 (ruff enforces, line-length 88)
- Meaningful names, small functions
- Fail fast and clearly — raise `BlitzIDError` subclasses, never bare
  `Exception`
- Prefer standard library
- No clever tricks
- Docstrings for public APIs (Google style: Args / Returns)
- No unnecessary inline comments
- Use uv (never pip / system python) for dev workflows
- No mutable module-level globals
- Private helpers use a `_` prefix (modules `_image.py`, `_models.py`
  and `_`-functions/methods); the public API is only what
  `src/blitzid/__init__.py` exports via `__all__`
- No dead code

## Style
Write the simplest correct solution. Delete anything that isn't needed.

## Conventions
- Public API lives in `src/blitzid/__init__.py` — every new public
  symbol goes there, into `__all__`, and into the README API table.
  Backward-compat aliases also live there (see the old exception names).
- Layout: two domain subpackages over shared infra. Face detection in
  `src/blitzid/face/` (`detector.py` — `FaceDetectorDNN` plus NMS /
  IoU / size filter; `_face.py` — `Face`, `DetectionMetrics`, cache,
  drawing; `_scrfd.py` — SCRFD preprocess/decode). Document reading in
  `src/blitzid/reading/` (`ocr.py`, `mrz.py`, `barcode.py`,
  `structurize.py`, `consistency.py`, `document.py`). Shared helpers at the
  top level: `_image.py` (input loading and normalization), `_models.py`
  (`ModelManager` — SCRFD ONNX download + CPU
  `onnxruntime.InferenceSession`, default cache dir via `platformdirs`).
  Subpackage `__init__.py` files hold no re-exports — the only public
  import path is the package root.
- Exceptions in `src/blitzid/exceptions.py`: `BlitzIDError` base,
  `ModelError` (model download/load), `ImageError` (load/validate/
  process), `MRZError` (MRZ not found / malformed / failed check-digit
  validation), `BarcodeError` (PDF417 not found / malformed / failed
  AAMVA validation), `FaceVerificationError` (no face for
  verification).
- Optional deps (PIL, rapidocr via the `ocr` extra, zxing-cpp via the
  `barcode` extra) are lazy-imported inside functions, guarded, with
  targeted `# type: ignore` codes; core must work without them.
- The MRZ layer (`src/blitzid/reading/mrz.py`) is pure ICAO 9303 logic
  over the OCR reader's text lines — no new models, no guessy repair:
  obvious OCR confusions (0/O, 1/I, 2/Z, 5/S, 6/G, 8/B) resolve
  deterministically by each field's alphabet, check digits and
  letter-only field validation gate everything else, and a failing
  field raises `MRZError`.
- Images normalize to 3-channel BGR `np.ndarray`; `ImageInput` is
  `path | ndarray | PIL Image`. SCRFD runs letterboxed at `det_size`
  (default 640×640, multiples of 32); boxes map back to original image
  coordinates. Model weights download on demand to the default models
  dir — `BLITZID_MODELS_DIR` when set, else the `platformdirs` user
  cache — never into the repo. `scripts/download_models.py` pre-fetches
  all weights for Docker images.
- No fallbacks between models or providers: exactly one SCRFD model on
  the explicit `CPUExecutionProvider`; anything missing or unloadable
  raises `ModelError`.
- Public detector surface: `detect_face`, `detect_face_landmarks`,
  `detect_face_with_metrics`, `extract_faces`, `visualize_detections`,
  `detect_faces_batch`. Landmarks follow SCRFD order (right eye, left
  eye, nose tip, right mouth corner, left mouth corner). The OCR
  surface is `RapidOCRReader.read()` returning `OCRText` records — one
  engine (RapidOCR on onnxruntime CPU), lazy-imported from the `ocr`
  extra. The MRZ surface is `MRZReader.read()` returning an
  `MRZRecord` (same `ocr` extra). The barcode surface is
  `BarcodeReader.read()` returning a `BarcodeRecord` (`barcode`
  extra, zxing-cpp decoder; the parser validates the AAMVA header's
  subfile offsets/lengths/count, gates mandatory fields per
  edition-pinned tag maps, keeps unknown tags raw in `extra_tags`,
  and raises `BarcodeError` naming the offending field).
- `scripts/` are demos, diagnostics, and dataset benchmarks — not part
  of the shipped package; they may use the `scripts` extra.
- Tests go in `tests/`; validation tests must run without model
  downloads or network.
- Update `CHANGELOG.md` under `[Unreleased]` for user-visible changes.
- Write 3.12-compatible code (`from __future__ import annotations`,
  `X | Y` unions); CI also runs 3.13 and 3.14.
