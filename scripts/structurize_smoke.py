"""StructuredOCRReader smoke test — calls the live OpenAI-compatible endpoint.

Exercises the full pydantic-ai structured-extraction path against the
configured LLM endpoint (`BLITZID_LLM_BASE_URL`, `BLITZID_LLM_API_KEY`,
and `BLITZID_LLM_MODEL` must be set) on three document kinds: hardcoded
sample MRZ lines, a specimen ID image, and a specimen license-plate
image. This is a smoke test: it makes network calls and asserts on the
returned records, so it is run on demand, not as part of the pytest
suite (which must never call the endpoint).

Run example::

    uv run python scripts/structurize_smoke.py

Exits non-zero if the endpoint is unreachable, misclassifies a document
kind, or returns no structured fields.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from datetime import date

from blitzid import OCRText, RapidOCRReader, StructuredOCR, StructuredOCRReader
from blitzid.reading.structurize import ReadKind

_MRZ_LINES = (
    OCRText(
        bbox=(0, 10, 440, 10),
        text="P<UTOERIKSSON<<ANNA<MARIA".ljust(44, "<"),
        confidence=0.9,
    ),
    OCRText(
        bbox=(0, 30, 440, 10),
        text="L898902C36UTO7408122F1204159ZE184226B<<<<<10",
        confidence=0.9,
    ),
)

_ID_IMAGE = "images/bub_der_personalausweis_kopie.jpg"
_PLATE_IMAGE = "images/ny_license_plate_wrap.jpg"

_PASSES: Sequence[tuple[ReadKind, Sequence[OCRText] | str]] = (
    ("mrz", _MRZ_LINES),
    ("id", _ID_IMAGE),
    ("plate", _PLATE_IMAGE),
)


def _report(record: StructuredOCR, expected_type: str, label: str) -> None:
    assert isinstance(record, StructuredOCR), type(record)
    assert record.document_type == expected_type, (
        f"{label}: expected document_type {expected_type!r}, "
        f"got {record.document_type!r}"
    )
    assert record.fields, f"{label}: no fields extracted"
    expected_fields: dict[str, dict[str, str | date]] = {
        "mrz": {
            "document_code": "P<",
            "issuer": "UTO",
            "document_number": "L898902C3",
            "nationality": "UTO",
            "surname": "ERIKSSON",
            "given_names": "ANNA MARIA",
            "birth_date": date(1974, 8, 12),
            "sex": "F",
            "expiry_date": date(2012, 4, 15),
        },
        "id": {
            "surname": "MUSTERMANN",
            "given_names": "ERIKA",
            "document_number": "LZ6311T47",
            "date_of_birth": date(1983, 8, 12),
        },
        "plate": {"plate_number": "HCM6223", "issuing_region": "NY"},
    }
    expected = expected_fields[expected_type]
    values = {field.name: field.value for field in record.fields}
    assert all(values.get(name) == value for name, value in expected.items()), (
        f"{label}: expected {expected}, got {values}"
    )
    print(f"  document_type: {record.document_type}")
    for field in record.fields:
        print(f"  {field.name}: {field.value} (conf={field.confidence:.2f})")


def main() -> None:
    """Run the smoke test and exit non-zero on failure."""
    if not __debug__:
        raise AssertionError("Run smoke tests without Python optimization (-O).")
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
        print("  auto-classification:")
        _report(auto, expected_type, f"{label} (auto)")

    print("Smoke test passed.")


if __name__ == "__main__":
    main()
