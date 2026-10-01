"""StructuredOCRReader smoke test — calls the live OpenAI-compatible endpoint.

Exercises the full pydantic-ai structured-extraction path against the
configured LLM endpoint (default a local vLLM server) on three document
kinds: hardcoded sample MRZ lines, a specimen ID image, and a specimen
license-plate image. This is a smoke test: it makes network calls and
asserts on the returned records, so it is run on demand, not as part of
the pytest suite (which must never call the endpoint).

Run example::

    uv run python scripts/structurize_smoke.py

Exits non-zero if the endpoint is unreachable, misclassifies a document
kind, or returns no structured fields.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence

from blitzid import OCRText, RapidOCRReader, StructuredOCR, StructuredOCRReader
from blitzid.reading.structurize import ReadKind

_MRZ_LINES = [
    OCRText(bbox=(0, 10, 10, 10), text="P<UTODBURTON<<JAMES", confidence=0.9),
    OCRText(bbox=(0, 30, 10, 10), text="L898902C<3UTO7408122F1204159", confidence=0.9),
]

_ID_IMAGE = "images/bub_der_personalausweis_kopie.jpg"
_PLATE_IMAGE = "images/ny_license_plate_wrap.jpg"

_PASSES: Sequence[tuple[ReadKind, Sequence[OCRText] | str]] = [
    ("mrz", _MRZ_LINES),
    ("id", _ID_IMAGE),
    ("plate", _PLATE_IMAGE),
]


def _report(record: StructuredOCR, expected_type: str, label: str) -> None:
    assert isinstance(record, StructuredOCR), type(record)
    assert record.document_type == expected_type, (
        f"{label}: expected document_type {expected_type!r}, "
        f"got {record.document_type!r}"
    )
    assert record.fields, f"{label}: no fields extracted"
    print(f"  document_type: {record.document_type}")
    for field in record.fields:
        print(f"  {field.name}: {field.value} (conf={field.confidence:.2f})")


def main() -> None:
    """Run the smoke test and exit non-zero on failure."""
    logging.basicConfig(level=logging.WARNING)

    reader = StructuredOCRReader(log_level=logging.WARNING)
    ocr = RapidOCRReader(log_level=logging.WARNING)

    for expected_type, source in _PASSES:
        label = source if isinstance(source, str) else "sample MRZ lines"
        print(f"— {label} —")
        texts = ocr.read(source) if isinstance(source, str) else source
        print(f"({len(texts)} OCR lines)")
        record = reader.read(texts, kind=expected_type)
        _report(record, expected_type, label)
        auto = reader.read(texts)
        print(
            f"  (auto-classified as {auto.document_type!r}, {len(auto.fields)} fields)"
        )

    print("Smoke test passed.")


if __name__ == "__main__":
    main()
