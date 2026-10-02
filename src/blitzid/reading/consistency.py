"""Cross-check printed document fields against the machine-readable zone.

Every identity document stores its data twice: once as printed text
(the ``structured`` section's view of it) and once in the code strip at
the bottom, the lines of letters and ``<`` built for machines (the
``mrz`` section, whose check digits are validated on parse). This
module compares the two copies field by field and reports where they
disagree. A disagreement is either a camera misread or an altered
document, and both deserve a second look. The code strip is the
trustworthy copy: the printed side has been through OCR and a model,
and every row keeps its value and confidence next to the code strip's.

The report also carries photo-vs-document rows: the document portrait's
predicted age band and gender against the document's birth date and sex
marker. Those rows are advisory by nature (a coarse model over one
photo), so a disagreement means retake or human review, never an
automatic rejection. Nothing compares race: documents do not carry it.

Pure functions over already-parsed records: no models, no I/O.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from typing import Literal

from blitzid._dates import parse_date
from blitzid.face._attributes import FaceAttributes
from blitzid.reading.mrz import MRZRecord
from blitzid.reading.structurize import StructuredOCR

__all__ = [
    "ConsistencyReport",
    "FieldComparison",
    "PhotoComparison",
    "cross_check",
]

Verdict = Literal["match", "partial", "mismatch", "unavailable"]

_COMPARED_FIELDS = (
    "document_number",
    "date_of_birth",
    "date_of_expiry",
    "surname",
    "given_names",
    "sex",
)

_PRINTED_ALIASES: dict[str, tuple[str, ...]] = {
    "date_of_birth": ("date_of_birth", "birth_date"),
    "date_of_expiry": ("date_of_expiry", "expiry_date"),
}

_MRZ_FIELDS: dict[str, str] = {
    "document_number": "document_number",
    "date_of_birth": "birth_date",
    "date_of_expiry": "expiry_date",
    "surname": "surname",
    "given_names": "given_names",
    "sex": "sex",
}

_NAME_FIELDS = frozenset({"surname", "given_names"})
_DATE_FIELDS = frozenset({"date_of_birth", "date_of_expiry"})

_BAND_TOLERANCE = 10
_PHOTO_SEXES = {"Male": "M", "Female": "F"}


def _photo_rows(
    photo: FaceAttributes | None,
    mrz: MRZRecord,
    printed: dict[str, tuple[str | None, float | None]],
    today: date,
) -> list[PhotoComparison]:
    """Build the photo-vs-document rows (age and sex)."""
    if photo is None:
        return [
            PhotoComparison("age", "unavailable", None, None, None),
            PhotoComparison("sex", "unavailable", None, None, None),
        ]
    birth = _document_birth(mrz, printed, today)
    age = None if birth is None else _age_in_years(birth, today)
    document_sex = getattr(mrz, _MRZ_FIELDS["sex"]) or printed["sex"][0]
    return [
        PhotoComparison(
            field="age",
            verdict="unavailable" if age is None else _age_verdict(age, photo.age),
            document_value=None if age is None else str(age),
            photo_value=photo.age,
            photo_confidence=photo.age_confidence,
        ),
        PhotoComparison(
            field="sex",
            verdict=_sex_verdict(document_sex, photo.gender),
            document_value=document_sex,
            photo_value=photo.gender,
            photo_confidence=photo.gender_confidence,
        ),
    ]


def _document_birth(
    mrz: MRZRecord,
    printed: dict[str, tuple[str | None, float | None]],
    today: date,
) -> date | None:
    """The document's birth date: the code strip's, else the printed one."""
    birth = _mrz_date(
        getattr(mrz, _MRZ_FIELDS["date_of_birth"]), expiry=False, today=today
    )
    if birth is not None:
        return birth
    raw = printed["date_of_birth"][0]
    return _printed_date(raw) if raw else None


def _age_in_years(birth: date, today: date) -> int:
    return (
        today.year - birth.year - ((today.month, today.day) < (birth.month, birth.day))
    )


def _age_verdict(age: int, band: str) -> Verdict:
    """Compare an exact age to a predicted band, one band of slack.

    The slack absorbs the model's coarseness and the fact that document
    portraits age at issue time: a ten-year-old passport legitimately
    shows a younger face than today's birth date implies.
    """
    bounds = _band_bounds(band)
    if bounds is None:
        return "unavailable"
    low, high = bounds
    if low <= age <= high:
        return "match"
    if low - _BAND_TOLERANCE <= age <= high + _BAND_TOLERANCE:
        return "partial"
    return "mismatch"


def _band_bounds(band: str) -> tuple[int, int] | None:
    """Map an age band like ``"20-29"`` or ``"70+"`` to inclusive bounds."""
    if band.endswith("+"):
        head = band[:-1]
        return (int(head), 200) if head.isdigit() else None
    low, sep, high = band.partition("-")
    if sep and low.isdigit() and high.isdigit():
        return int(low), int(high)
    return None


def _sex_verdict(document_sex: str | None, gender: str | None) -> Verdict:
    """Compare the document's sex marker to the predicted gender."""
    if not document_sex or not gender:
        return "unavailable"
    if document_sex == "X":
        return "partial"
    expected = _PHOTO_SEXES.get(gender)
    if expected is None:
        return "unavailable"
    return "match" if document_sex == expected else "mismatch"


@dataclass(frozen=True)
class FieldComparison:
    """One field's agreement between the two on-document copies.

    Attributes:
        field: Compared field name, e.g. ``"date_of_birth"``.
        verdict: ``"match"``, ``"partial"`` (names share only some
            words), ``"mismatch"``, or ``"unavailable"`` (a side holds
            no usable value, so nothing was compared).
        mrz_value: The code strip's raw value (the trustworthy copy).
        printed_value: The printed side's value.
        printed_confidence: The model's confidence for the printed
            value.
    """

    field: str
    verdict: Verdict
    mrz_value: str | None
    printed_value: str | None
    printed_confidence: float | None


@dataclass(frozen=True)
class PhotoComparison:
    """One photo attribute checked against the document's data.

    Attributes:
        field: ``"age"`` or ``"sex"``.
        verdict: ``"match"``, ``"partial"`` (close to the document,
            within the model's tolerance), ``"mismatch"``, or
            ``"unavailable"`` (a side holds no usable value).
        document_value: The document's value — exact age in years for
            ``"age"``, the sex marker (M/F/X) for ``"sex"``. This is
            the trustworthy side.
        photo_value: The photo model's label (``"20-29"``, ``"Female"``).
        photo_confidence: The photo model's softmax score for it.
    """

    field: str
    verdict: Verdict
    document_value: str | None
    photo_value: str | None
    photo_confidence: float | None


@dataclass(frozen=True)
class ConsistencyReport:
    """Comparison of the on-document copies and of the portrait photo.

    Attributes:
        comparisons: One row per compared field between the printed
            text and the code strip, always all of them.
        photo_comparisons: The photo-vs-document rows (age, sex),
            always both of them; ``unavailable`` when no photo
            attributes were supplied.
        consistent: False when any row in either block is a mismatch.
            Partial matches stay visible in the rows but leave this
            flag true.
    """

    comparisons: tuple[FieldComparison, ...]
    photo_comparisons: tuple[PhotoComparison, ...]
    consistent: bool


def cross_check(
    structured: StructuredOCR,
    mrz: MRZRecord,
    photo: FaceAttributes | None = None,
    today: date | None = None,
) -> ConsistencyReport:
    """Compare the printed fields and the portrait photo against the document.

    Args:
        structured: The printed side, as extracted from the document.
        mrz: The code-strip record (check digits validated).
        photo: The document portrait's predicted attributes (the
            highest-scoring face), when available.
        today: Reference date for the birth-century pivot of raw
            ``YYMMDD`` values and for exact ages. Defaults to the
            current local date.

    Returns:
        The per-field report. ``consistent`` is False when any field
        in either block disagrees outright.
    """
    today = today or date.today()
    printed = _printed_values(structured)
    rows = []
    for field in _COMPARED_FIELDS:
        mrz_value = getattr(mrz, _MRZ_FIELDS[field]) or None
        printed_value, confidence = printed[field]
        if field in _DATE_FIELDS:
            verdict = _date_verdict(mrz_value, printed_value, field, today)
        elif field in _NAME_FIELDS:
            verdict = _name_verdict(mrz_value, printed_value)
        else:
            verdict = _text_verdict(mrz_value, printed_value)
        rows.append(
            FieldComparison(
                field=field,
                verdict=verdict,
                mrz_value=mrz_value,
                printed_value=printed_value,
                printed_confidence=confidence,
            )
        )
    photo_rows = _photo_rows(photo, mrz, printed, today)
    clean = all(row.verdict != "mismatch" for row in rows) and all(
        row.verdict != "mismatch" for row in photo_rows
    )
    return ConsistencyReport(
        comparisons=tuple(rows),
        photo_comparisons=tuple(photo_rows),
        consistent=clean,
    )


def _printed_values(
    structured: StructuredOCR,
) -> dict[str, tuple[str | None, float | None]]:
    """Map each compared field to the printed side's ``(value, confidence)``."""
    values: dict[str, tuple[str | None, float | None]] = {}
    for field in _COMPARED_FIELDS:
        hit = next(
            (
                item
                for alias in _PRINTED_ALIASES.get(field, (field,))
                for item in structured.fields
                if item.name == alias
            ),
            None,
        )
        if hit is None:
            values[field] = (None, None)
            continue
        raw = hit.value.isoformat() if isinstance(hit.value, date) else hit.value
        values[field] = (raw or None, hit.confidence)
    return values


def _date_verdict(
    mrz_value: str | None, printed_value: str | None, field: str, today: date
) -> Verdict:
    if not mrz_value or not printed_value:
        return "unavailable"
    left = _mrz_date(mrz_value, expiry=field == "date_of_expiry", today=today)
    right = _printed_date(printed_value)
    if left is None or right is None:
        return "unavailable"
    return "match" if left == right else "mismatch"


def _name_verdict(mrz_value: str | None, printed_value: str | None) -> Verdict:
    if not mrz_value or not printed_value:
        return "unavailable"
    left, right = _words(mrz_value), _words(printed_value)
    if not left or not right:
        return "unavailable"
    if left == right:
        return "match"
    if left <= right or right <= left:
        return "partial"
    return "mismatch"


def _text_verdict(mrz_value: str | None, printed_value: str | None) -> Verdict:
    if not mrz_value or not printed_value:
        return "unavailable"
    return "match" if _norm(mrz_value) == _norm(printed_value) else "mismatch"


def _mrz_date(raw: str, *, expiry: bool, today: date) -> date | None:
    """Parse a raw ``YYMMDD`` code-strip date.

    Expiry dates land in the 2000s. Birth dates take the most recent
    century that does not put the date after *today*, the same rule the
    extraction prompts state.
    """
    if len(raw) != 6 or not raw.isdigit():
        return None
    yy, mm, dd = int(raw[:2]), int(raw[2:4]), int(raw[4:])
    try:
        parsed = date(2000 + yy, mm, dd)
    except ValueError:
        return None
    if not expiry and parsed > today:
        try:
            return date(1900 + yy, mm, dd)
        except ValueError:
            return None
    return parsed


def _printed_date(raw: str) -> date | None:
    parsed = parse_date(raw)
    return parsed if isinstance(parsed, date) else None


def _words(value: str) -> frozenset[str]:
    return frozenset(_norm(value).split())


def _norm(value: str) -> str:
    """Uppercase, collapse whitespace, and drop trailing filler marks."""
    return " ".join(value.upper().split()).strip().rstrip("<").strip()
