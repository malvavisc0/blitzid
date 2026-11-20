# Coding Standard — blitzid

Rules for contributors and coding agents. Priority: clean, simple,
maintainable code with no known defects. This file is the source of truth
for code, tests, and review.

## Agent contract

- Read the Agent contract, Definition of done, and Security/PII sections every
  time. Read the relevant source, tests, and sections below before editing.
  Follow existing patterns over generic preferences.
- Plan multi-file or behavior-changing work before editing. Make the smallest
  correct change; no drive-by refactors, dependency changes, public API
  changes, or reformatting outside the task.
- Verify with the commands in [Definition of done](#definition-of-done). Never
  claim a check passed unless you ran it. Report the exact command and result.
- When a check fails, fix the cause. Never weaken or delete a test, add a broad
  `# type: ignore`, or suppress a lint/security rule to go green.
- Stop and ask only for genuine blockers: ambiguous requirements, public
  API/behavior changes, new dependencies, destructive migrations, conflicting
  instructions, or security-sensitive changes.
- Never commit, amend, push, or open a PR unless explicitly asked. Never commit
  secrets, PII, model weights, caches, or generated files.
- Keep the diff reviewable. Do not modify unrelated code or tests.
- When a bug is reported, add a failing regression test first, then fix it.
  When the same mistake recurs, update this file.
- Finish with a short report: what changed, commands run and results, known
  limitations, and anything not done. No status claims without evidence.

### Communication

Write for a senior reader. Be terse: results, not intentions. No preamble,
recap, hand-holding, or "here's what I did". Skip explanations unless asked.
Never narrate tool use. Answer and move on. Speak plainly, like a human.

## Project

`blitzid` — modular DNN-based ID document reading framework, optimized for
CPU. Typed Python library (`blitzid`, src layout) with two layers:

- **Face** (`src/blitzid/face/`): YuNet detector (MIT) + SFace recognizer
  (Apache 2.0) via OpenCV's CPU DNN backend. LRU result cache, multi-scale
  detection, NMS + size filtering, batch processing, bbox visualization, and
  1:1 cosine verification. FairFace attributes and MiniFASNet anti-spoofing
  use onnxruntime CPU. `_motion.py` adds challenge-response liveness.
- **Reading** (`src/blitzid/reading/`): RapidOCR text lines (core), ICAO 9303
  MRZ parsing (TD1/TD2/TD3), AAMVA PDF417 driver's-license parsing (core,
  `zxing-cpp`), LLM-backed structured extraction (core), document
  localization/perspective crop/QC, and printed-vs-MRZ-vs-photo consistency.

Inputs are file paths, NumPy arrays (BGR), or PIL Images. Face outputs are
`(x, y, w, h, confidence)` tuples + crops.

## Stack

- Python 3.12+ (CI: 3.12/3.13/3.14), uv, pyproject.toml (hatchling backend,
  src layout, `py.typed`).
- Core deps: numpy, onnxruntime (CPU), opencv-python-headless>=5,
  platformdirs, pydantic, rapidocr, pydantic-ai, httpx2, langfuse, zxing-cpp,
  omegaconf.
- Extras: `api` (FastAPI service), `scripts` (demo/benchmark tools), `dev`
  (tests, lint, types, API test deps). See `pyproject.toml` for exact lists.
- Tooling: mypy strict; ruff (line-length 88, target py312; E, W, F, I, UP, B,
  SIM, RUF); pytest; pre-commit.
- Use uv for all Python development workflows; never pip or system Python for
  project work.

## Definition of done

Run from the repo root after every change:

```bash
uv sync --extra dev
uv run ruff check --fix src/ tests/ scripts/
uv run ruff format src/ tests/ scripts/
uv run pre-commit run --all-files
uv run mypy src/ scripts/
uv run pytest tests/ -v
```

CI runs `ruff check`, `ruff format --check`, and `mypy`; `pytest` runs on
Python 3.12/3.13/3.14. This happens for PRs and pushes to `master`. Match it
locally and never leave a change red.

Complexity and dead code are manual gates:

```bash
uvx radon cc src scripts tests -s | grep -E '\-\s(C|D|E|F)\s'   # must be empty
uvx vulture src/ tests/ scripts/ --min-confidence 60            # review every hit
```

Security scans (adopt as CI gates when ready):

```bash
uvx bandit -r src scripts
uv run --with pip-audit pip-audit
```

`B101` hits in `scripts/*_smoke.py` are expected: smoke scripts use asserts as
checks. Do not suppress `B101` in `src`.

Checklist:

- [ ] New behavior has tests, including failure paths.
- [ ] Every bug fix has a regression test that fails without the fix.
- [ ] Errors are specific, chained, and contain no PII or secrets.
- [ ] Untrusted input is validated; verification and security checks fail
      closed.
- [ ] Complexity is A/B, there is no dead code, and there are no unjustified
      suppressions.
- [ ] Public API changes update `src/blitzid/__init__.py`, `__all__`, and the
      README API reference.
- [ ] User-visible changes update `CHANGELOG.md` under `[Unreleased]`.
- [ ] `git diff` contains only intended changes and no PII, secrets, weights,
      caches, or generated files.

## Code rules

### Design

- DRY, KISS, YAGNI. Write the simplest correct solution; delete what is not
  needed.
- Small functions with one purpose; pure functions where possible. Keep I/O at
  the edges and business logic testable without files, network, or models.
- Prefer immutable records: frozen dataclasses, tuples, enums. Mutability is
  for caches and explicit state, not shared module data.
- Inject dependencies through constructors/parameters. No hidden globals, no
  import-time side effects, no mutable module-level state. Constants and
  `__all__` are exempt.
- No mutable default arguments.
- Parse, don't validate: validate and normalize untrusted input once at the
  boundary, then trust typed data inside. Do not scatter defensive checks.
- Explicit over implicit. No clever tricks. No speculative abstractions.
- Fail fast and clearly: raise `BlitzIDError` subclasses, never bare
  `Exception`. No fallbacks that hide missing models, config, or bad input.
- Prefer the standard library. Add a dependency only when it earns its place.
- Environment-variable names are configuration keys, not values. Use a
  descriptive `*_ENV` constant only when it is reused (errors/tests); never
  write `X_ENV = "X"`. For one-off lookups, pass the literal to
  `os.environ.get`.
- Use context managers for files, sessions, and other resources. Prefer
  `pathlib` over string path manipulation. Use `logging`, not `print`.
- Shared instances must be thread-safe: immutable state or explicit locks.
  Document the thread-safety contract.
- No runtime `assert` for validation; use explicit errors. `assert` is for
  tests and type narrowing.

### Typing

- Type hints always; mypy strict must pass. No untyped definitions, no
  incomplete signatures. Avoid `Any` except at unavoidable third-party
  boundaries.
- Write 3.12-compatible code: PEP 604 unions (`X | Y`) and built-in generics
  (`list[str]`, `dict[str, int]`). Use `from __future__ import annotations` in
  new modules, except re-export-only `__init__.py` files.
- Make illegal states unrepresentable with enums, literals, and validated
  Pydantic/dataclass models.
- Targeted `# type: ignore[code]` only at third-party boundaries, with a
  reason. Put stub overrides in `pyproject.toml`; never add broad ignores.
- Use Google-style docstrings with the existing annotations. Document meaning
  and contracts; avoid restating annotation types unless the meaning is
  non-obvious.

### Errors

- Package errors live in `src/blitzid/exceptions.py`: `BlitzIDError` (base),
  `ModelError` (download/load), `ImageError` (load/validate/process), `MRZError`
  (not found / malformed / failed check digit), `BarcodeError` (PDF417 not
  found / malformed / failed AAMVA validation), `FaceVerificationError` (no
  face to verify).
- Raise the most specific package error. Never raise bare `Exception` or leak
  OpenCV/onnxruntime/pydantic errors.
- Catch only exceptions you can handle; chain with `raise ... from exc`. Keep
  `try` blocks minimal. No silent `except: pass`.
- Messages state what failed and what to do next (fix input, install extra,
  set env var). Never include secrets or personal data.
- Use `None` only for a documented absence; never return sentinel values
  (`-1`, empty string, `False`) to signal an error.
- Fail closed for verification, liveness, and security decisions. Malformed
  or non-finite model output is never a success.
- Missing or unloadable model weights → `ModelError`. Missing core
  dependencies → `BlitzIDError` with an install hint.

### Imports and dependencies

- Heavy/optional imports (PIL, rapidocr, pydantic-ai/httpx2, langfuse,
  zxing-cpp) are lazy-imported inside functions and guarded. A missing core
  dependency is a broken install; PIL remains optional.
- Extras (`fastapi`, `redis`, ...) are imported only inside
  `src/blitzid/api/`; core modules must not require them.
- No circular imports. Use `TYPE_CHECKING` for type-only imports.
- Keep `uv.lock` current. New dependencies need a justification: license,
  maintenance, security, CPU footprint, and the alternative considered.

### Complexity

- Every function/method ranks A or B (score ≤10), for the whole codebase. A
  pre-existing C/D/E/F block is a defect: fix it when you encounter it, never
  work around it or leave it for later.
- When a change would push a block to C+, split it into smaller functions
  (early returns, extracted helpers, dispatch tables) instead of adding
  branches. Fix existing C+ blocks in the same pass.
- Prefer ≤3 nesting levels. If a function needs a block comment to explain
  what a section does, extract that section.
- Verify with radon; it must produce no C–F entries.

### Dead code

- No dead code: never-read fields, zero-call-site functions, unused imports,
  stale aliases, `_`-prefixed helpers with no callers. Delete in the same pass
  you notice it.
- Static-analysis exceptions: `@pytest.fixture`, mock attributes
  (`return_value` etc.), framework hooks vulture cannot see (FastAPI route
  handlers, Pydantic fields/validators), and optional-dependency shims marked
  `# pragma: no cover`.
- Review vulture hits; don't delete a framework hook just because vulture
  cannot see it. Everything else must have a call site.

### Docs and comments

- Google-style docstrings on public APIs: summary, Args, Returns, Raises.
- No unnecessary inline comments. Comment non-obvious constraints or
  rationale, not what the code does. No decorative section banners, no
  restating the test name (`# should not raise`), and no narrating the next
  line (`# create image`).
- Public behavior changes update the README API reference and
  `CHANGELOG.md`; changed rules update this file.

### Style

- PEP 8 via ruff, line length 88. Meaningful names, small functions.
  `snake_case` functions/variables, `PascalCase` classes, `UPPER_SNAKE_CASE`
  constants.
- Private helpers use a `_` prefix (modules `_image.py`, `_models.py`,
  `_`-functions/methods).
- Use f-strings for messages; use lazy `%`-args in logging calls.
- No wildcard imports, commented-out code, or trailing whitespace.

## Layout and public API

- Public API lives in `src/blitzid/__init__.py`: every new public symbol goes
  there, into `__all__`, and into the README API reference. Backward-compat
  aliases also live there.
- Layout: two domain subpackages over shared infra.
  - Face (`src/blitzid/face/`): `detector.py` (`FaceDetectorDNN` plus NMS/IoU/
    size filter), `_face.py` (`Face`, `DetectionMetrics`, cache, drawing),
    `_yunet.py` (mapping/letterbox), `_sface.py` (alignment/embedding),
    `verifier.py` (`FaceVerifier`), `_attributes.py` (FairFace),
    `_antispoof.py` (MiniFASNet), `_motion.py` (challenge-response liveness).
  - Reading (`src/blitzid/reading/`): `ocr.py`, `mrz.py`, `barcode.py`,
    `structurize.py`, `consistency.py`, `document.py`, `_observability.py`.
  - Shared top level: `_image.py` (input loading/normalization), `_models.py`
    (`ModelManager` — pinned download + CPU `onnxruntime.InferenceSession`,
    platformdirs cache), `_dates.py` (date coercion).
  - Subpackage `__init__.py` files hold no re-exports; the only public import
    path is the package root.
- The optional HTTP API lives in `src/blitzid/api/` behind the `api` extra;
  core modules must not import it.
- `scripts/` are demos, diagnostics, and dataset benchmarks — not part of the
  shipped package; they may use the `scripts` extra. Tests go in `tests/`.
- Public surfaces:
  - Face: `detect_face`, `detect_face_landmarks`, `detect_face_with_metrics`,
    `extract_faces`, `visualize_detections`, `detect_faces_batch`,
    `FaceVerifier`. Landmarks follow YuNet order (right eye, left eye, nose
    tip, right mouth corner, left mouth corner).
  - OCR: `RapidOCRReader.read()` → `OCRText` records. One engine (RapidOCR on
    onnxruntime CPU), lazy-imported from the core dependencies.
  - MRZ: `MRZReader.read()` → `MRZRecord`.
  - Barcode: `BarcodeReader.read()` → `BarcodeRecord`. The zxing-cpp decoder
    is core; the parser validates the AAMVA header's subfile
    offsets/lengths/count, gates mandatory fields per edition-pinned tag maps,
    keeps unknown tags raw in `extra_tags`, and raises `BarcodeError` naming
    the offending field.
  - Consistency: `cross_check` → `ConsistencyReport`.
  - Document: `DocumentCropper.crop()` → `(crop, QualityReport)`.

## Domain conventions

- MRZ (`src/blitzid/reading/mrz.py`) is pure ICAO 9303 logic over the OCR
  reader's text lines — no new models, no guessy repair. Obvious OCR
  confusions (0/O, 1/I, 2/Z, 5/S, 6/G, 8/B) resolve deterministically by each
  field's alphabet; check digits and letter-only field validation gate
  everything else; a failing field raises `MRZError`.
- Images normalize to 3-channel BGR `np.ndarray`; `ImageInput` is
  `path | ndarray | PIL Image`. YuNet runs letterboxed at `det_size` (default
  640×640, multiples of 32); boxes map back to original image coordinates.
- Model weights download on demand to `BLITZID_MODELS_DIR` when set, else the
  `platformdirs` user cache — never into the repo. Downloads are pinned by
  SHA-256. `scripts/download_models.py` pre-fetches all weights for Docker.
- No fallbacks between models or providers: exactly one YuNet detector and one
  SFace recognizer on the CPU DNN backend; anything missing or unloadable
  raises `ModelError`.

## Testing

Zero bugs is not a guarantee any process can make. The target is **no known
defects**: behavior is specified by tests, every change is verified locally
and in CI, and every fixed bug gets a regression test. Coverage and mutation
scores are diagnostics, not proof of correctness.

- Use the test pyramid: many fast unit tests, integration tests at real
  boundaries (filesystem, Redis, HTTP, model I/O), few end-to-end tests. Do
  not over-mock: a mock only proves the simulation you wrote.
- Tests live in `tests/` and mirror module names (`test_mrz.py`). Run with
  `uv run pytest tests/ -v`.
- Unit tests must not require network, Redis, or a live LLM. Use stubs/mocks,
  `fakeredis`, and FastAPI `TestClient`.
- Model-backed tests may download weights on first use to the shared cache;
  for offline runs, point `BLITZID_MODELS_DIR` at a pre-populated cache.
  Live-engine tests are gated behind `BLITZID_OCR_INTEGRATION=1`.
- Follow Arrange–Act–Assert. One behavior per test; name tests for behavior
  and expected outcome, not implementation.
- Test the public contract, not private details. Test private helpers directly
  only when the logic is complex and the public path is awkward to exercise.
- Parametrize cases instead of duplicating tests. Put shared setup in
  `conftest.py` fixtures; keep fixture scopes deliberate; use `tmp_path` for
  files and `monkeypatch` for environment/attributes.
- Tests must be independent and order-insensitive. No shared mutable state
  between tests. Seed all randomness.
- Cover failure paths, not only the happy path: empty/malformed/oversized
  input, wrong types, missing files, corrupt images, invalid check digits,
  bad AAMVA offsets/counts, non-finite model output, timeouts, and
  concurrency/lease expiry.
- Use `pytest.raises(..., match=...)` for errors; assert the error type and
  the actionable part of the message.
- Add a regression test for every bug. Reproduce the failure first; the test
  must fail without the fix and pass with it.
- No unconditional skips or `xfail` tests without a linked issue and a reason.
  A skipped test is not coverage.
- Avoid sleeps except where lease/TTL semantics require them. Keep them short,
  deterministic, and never use a retry loop to hide a race. Flaky tests are
  bugs: fix the root cause, never rerun until green.
- Property-based tests are high-value for pure logic: MRZ round trips and
  check-digit properties, AAMVA offset/length invariants, bbox/letterbox
  mapping, NMS invariants, and date parsing. Add Hypothesis to the `dev` extra
  when adopting this.
- Mutation testing is a diagnostic for critical pure logic, not a whole-repo
  gate. Surviving mutants mean weak assertions; fix the tests, not the score.
- Coverage is a signal, not a target. 100% coverage does not prove correctness;
  do not add assertion-free tests to raise it.
- Test security behavior: malformed/untrusted input, no secrets or PII in
  logs/errors/traces, path traversal, and unsafe deserialization.
- Benchmark critical paths with `scripts/benchmark.py`; do not claim a
  performance improvement without measurement.

## Security and PII

Faces, IDs, and images of real people are biometric PII. Never use or commit
PII — not in code, tests, docs, plans, examples, or commit messages.

- Tests, docstrings, and examples use synthetic imagery (`np.zeros(...)`
  arrays, drawn shapes, invented filenames) or the committed fixtures in
  `images/`: official specimen documents and historical public-domain
  photographs of long-deceased persons, both attributed in `images/README.md`.
  Living people's photos stay out — a real person's photo in a test fixture is
  still that person's data.
- Real photos, ID scans, and captured frames stay in local, untracked
  artifacts — never `images/`, never the repo.
- Datasets (e.g. MIDV-500) are fetched at runtime into cache dirs by
  `scripts/`; never commit downloaded samples or model weights (`models/`
  weights are gitignored — only README/.gitkeep are tracked).
- Before committing anything touched by real imagery, check the diff:
  `git diff --stat` and open every added binary. A living person's face or a
  real document does not go in the repo.
- Treat images, OCR/MRZ text, barcodes, and LLM output as untrusted input.
  Bound sizes, reject malformed data, never `eval`/`exec`/`pickle` it, never
  use `shell=True`, and never build paths from untrusted text.
- Never log raw OCR text, MRZ fields, names, document numbers, or face
  embeddings. Log counts, types, and opaque IDs. Redact file paths when they
  are user-sensitive.
- Secrets come from the environment (`.env` is untracked). Never hardcode or
  log them. LLM and telemetry integrations are opt-in; do not send PII to an
  external service by default.
- Prompt injection is an input-validation problem: OCR/document text may be
  adversarial. It must not change tool permissions, system instructions, or
  safety checks; structured output is validated before use.

## Commits

- One-line messages only. No body, no trailers.
- Stage only intended files. Never commit secrets, PII, model weights, caches,
  or generated artifacts.

## References

- [PEP 8 — Style Guide for Python Code](https://peps.python.org/pep-0008/)
- [PEP 20 — The Zen of Python](https://peps.python.org/pep-0020/)
- [Google Python Style Guide](https://google.github.io/styleguide/pyguide.html)
- [mypy strict mode](https://mypy.readthedocs.io/en/stable/command_line.html#cmdoption-mypy-strict)
- [pytest — Good Integration Practices](https://docs.pytest.org/en/stable/explanation/goodpractices.html)
- [pytest — Fixtures](https://docs.pytest.org/en/stable/how-to/fixtures.html)
- [Hypothesis — property-based testing](https://hypothesis.readthedocs.io/en/latest/)
- [AGENTS.md — guidance for coding agents](https://agents.md/)
- [Bandit — Python security linting](https://bandit.readthedocs.io/en/latest/)
- [pip-audit — dependency vulnerability scanning](https://pypi.org/project/pip-audit/)
- [The Practical Test Pyramid](https://martinfowler.com/articles/practical-test-pyramid.html)
