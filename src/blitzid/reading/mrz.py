"""MRZReader — ICAO 9303 machine-readable zone reading.

Selects MRZ lines from OCR text, validates them via ICAO 9303 check
digits plus the letter-only fields (document code, issuer, nationality),
and parses TD1 (3x30, ID cards), TD2 (2x36), and TD3 (2x44, passports)
layouts into :class:`MRZRecord` fields. Runs on top of the ``ocr``
extra's RapidOCRReader. Obvious OCR confusions are resolved before
validation: each field's alphabet determines the character (0/O, 1/I,
2/Z, 5/S, 6/G, 8/B), and the check digits then gate the result, so a
wrong guess still raises. A zone that does not validate raises
:class:`~blitzid.exceptions.MRZError` with the failing field; nothing
is invented or partially repaired. One check digit is never required:
ICAO fills the document-number check position with ``"<"`` for numbers
longer than the field (overflow in the optional-data field); the
composite check digit still covers the full number.
"""

from __future__ import annotations

import logging
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime
from itertools import combinations
from types import MappingProxyType

from blitzid._image import ImageInput
from blitzid.exceptions import MRZError
from blitzid.reading.ocr import OCRText, RapidOCRReader

__all__ = ["MRZReader", "MRZRecord"]

_CHARSET = frozenset("ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789<")
_LETTERS = frozenset("ABCDEFGHIJKLMNOPQRSTUVWXYZ")
_WEIGHTS = (7, 3, 1)
_LINE_COUNT: Mapping[int, int] = MappingProxyType({30: 3, 36: 2, 44: 2})

# Obvious OCR confusions: one-to-one between digits and the letters
# they are misread from. A digit in a letters-only field can only be
# its paired letter, and the other way around.
_OCR_TO_LETTERS = str.maketrans("012586", "OIZSBG")
_OCR_TO_DIGITS = str.maketrans("OIZSBG", "012586")


@dataclass(frozen=True)
class MRZRecord:
    """Parsed ICAO 9303 machine-readable zone.

    Attributes:
        mrz_type: "TD1", "TD2", or "TD3".
        document_code: Two-character document code, e.g. "P<" or "I<".
        issuer: Three-character issuing state code.
        document_number: Document number, fillers stripped. Numbers
            longer than the field keep their overflow at the start of
            ``optional_data1`` (ICAO extended-number convention).
        birth_date: Birth date as YYMMDD.
        sex: "M", "F", or "X" (MRZ "<" normalized).
        expiry_date: Expiry date as YYMMDD.
        nationality: Three-character nationality code.
        surname: Primary identifier, fillers as spaces.
        given_names: Secondary identifier, fillers as spaces.
        optional_data1: Raw first optional field (TD3: personal number).
        optional_data2: Raw second optional field, empty outside TD1.
    """

    mrz_type: str
    document_code: str
    issuer: str
    document_number: str
    birth_date: str
    sex: str
    expiry_date: str
    nationality: str
    surname: str
    given_names: str
    optional_data1: str
    optional_data2: str = ""


def _char_value(ch: str) -> int:
    """ICAO 9303 character value: '<' is 0, digits 0-9, letters 10-35."""
    if ch == "<":
        return 0
    if ch.isdigit():
        return int(ch)
    return ord(ch) - ord("A") + 10


def _check_digit(text: str) -> int:
    """Compute the ICAO 9303 check digit (7-3-1 weighting, mod 10)."""
    return sum(_char_value(ch) * _WEIGHTS[idx % 3] for idx, ch in enumerate(text)) % 10


def _verify(label: str, text: str, expected: str) -> None:
    """Verify a printed check digit against the computed one.

    Args:
        label: Field name for the error message.
        text: The field the check digit covers.
        expected: The printed check digit character.

    Raises:
        MRZError: If the printed digit does not match the computed one.
    """
    if expected == "<" and not text.strip("<"):
        return
    digit = _fix_digits(expected)
    computed = _check_digit(text)
    if not digit.isdigit() or computed != int(digit):
        raise MRZError(
            f"{label} check digit mismatch: printed {expected!r}, computed {computed}"
        )


def _verify_document_number(label: str, text: str, expected: str) -> None:
    """Verify the printed document-number check digit.

    ICAO 9303: a number longer than its field puts the overflow at the
    start of the optional-data field and fills the check position with
    ``"<"`` — no check digit is printed, so none is required. The
    composite check digit still covers the full number.

    Raises:
        MRZError: If the printed digit does not match the computed one.
    """
    if expected != "<":
        _verify(label, text, expected)


def _fix_letters(code: str) -> str:
    """Undo digit confusions in a letters-only field (0/O, 1/I, ...)."""
    return code.translate(_OCR_TO_LETTERS)


def _fix_digits(code: str) -> str:
    """Undo letter confusions in a digits-only field (O/0, I/1, ...)."""
    return code.translate(_OCR_TO_DIGITS)


def _verify_letters(label: str, code: str) -> str:
    """Resolve and verify a letters-only MRZ field (fillers tolerated).

    A digit in a letters-only field (document code, issuer,
    nationality, names, sex) is an OCR confusion and can only be its
    paired letter (0/O, 1/I, 2/Z, 5/S, 6/G, 8/B), so it is mapped
    back first. Trailing '<' fillers are ignored, so "UTO" and "UT<"
    both pass; the field must still contain at least one letter.

    Returns:
        The normalized field.

    Raises:
        MRZError: If the normalized *code* contains a non-letter.
    """
    code = _fix_letters(code)
    letters = code.rstrip("<")
    if not letters or not all(ch in _LETTERS for ch in letters):
        raise MRZError(f"invalid {label}: {code!r}")
    return code


def _verify_date(label: str, yymmdd: str) -> str:
    """Resolve and verify a YYMMDD MRZ date is a real calendar date.

    Letters in a digits-only field are OCR confusions (O/0, I/1, ...)
    and are mapped back first.

    Returns:
        The normalized date field.

    Raises:
        MRZError: If the field is not a real calendar date.
    """
    yymmdd = _fix_digits(yymmdd)
    try:
        datetime.strptime(yymmdd, "%y%m%d")
    except ValueError as e:
        raise MRZError(f"invalid {label}: {yymmdd!r} is not a real date") from e
    return yymmdd


def _fillers_to_spaces(text: str) -> str:
    """Turn '<' fillers into single spaces, dropping empty parts."""
    return " ".join(part for part in text.split("<") if part)


def _parse_name(field: str) -> tuple[str, str]:
    """Split a name field into ``(surname, given_names)``."""
    surname, _, given = field.partition("<<")
    return _fillers_to_spaces(surname), _fillers_to_spaces(given)


def _normalize_sex(sex: str) -> str:
    """Map an MRZ sex character to "M", "F", or "X".

    Digit confusions (0/O, ...) are mapped back first.

    Raises:
        MRZError: If the character is not M, F, or "<".
    """
    sex = _fix_letters(sex)
    if sex not in {"M", "F", "<"}:
        raise MRZError(f"invalid sex character: {sex!r}")
    return "X" if sex == "<" else sex


def _is_mrz_line(text: str) -> bool:
    """Whether *text* could be one MRZ line (charset, length, content)."""
    return (
        len(text) in _LINE_COUNT
        and all(ch in _CHARSET for ch in text)
        and bool(text.strip("<"))
    )


def _select_mrz_lines(texts: list[OCRText]) -> Iterator[list[str]]:
    """Yield candidate MRZ line groups, bottom-most first.

    Candidates are lines of exactly 30, 36, or 44 characters over the
    MRZ charset, re-assembled top-to-bottom by bbox y (the reader sorts
    by confidence, not position). For each length, every combination
    of the required line count is yielded bottom-most first, so a
    stray OCR line above, between, or below the zone cannot displace
    real MRZ lines without a validation attempt.
    """
    candidates: list[tuple[int, str]] = []
    for text in texts:
        line = text.text.replace(" ", "").upper()
        if _is_mrz_line(line):
            candidates.append((text.bbox[1], line))
    candidates.sort()
    for length, count in _LINE_COUNT.items():
        group = [(y, text) for y, text in candidates if len(text) == length]
        for combo in combinations(reversed(group), count):
            yield [text for _, text in sorted(combo)]


def _read_mrz(texts: list[OCRText]) -> MRZRecord:
    """Parse the first candidate MRZ line group that validates.

    Raises:
        MRZError: If there is no candidate group or none validates.
    """
    errors: list[str] = []
    for lines in _select_mrz_lines(texts):
        try:
            return _parse(lines)
        except MRZError as e:
            errors.append(str(e))
    if errors:
        raise MRZError(f"no valid MRZ: {'; '.join(errors)}")
    raise MRZError("no MRZ found among OCR text lines")


def _parse(lines: Sequence[str]) -> MRZRecord:
    """Parse normalized MRZ lines, dispatching on the line layout.

    Raises:
        MRZError: If the layout is not a supported MRZ type.
    """
    sizes = [len(line) for line in lines]
    if sizes == [30, 30, 30]:
        return _parse_td1(lines[0], lines[1], lines[2])
    if sizes == [36, 36]:
        return _parse_td2(lines[0], lines[1])
    if sizes == [44, 44]:
        return _parse_td3(lines[0], lines[1])
    raise MRZError(f"unsupported MRZ line layout: {sizes}")


def _parse_td1(l1: str, l2: str, l3: str) -> MRZRecord:
    """Parse a TD1 (3x30) machine-readable zone."""
    code = _verify_letters("TD1 document code", l1[0:2])
    issuer = _verify_letters("TD1 issuer", l1[2:5])
    document_number = l1[5:14]
    doc_check = _fix_digits(l1[14])
    optional1 = l1[15:30]
    birth_date = _verify_date("TD1 birth date", l2[0:6])
    birth_check = _fix_digits(l2[6])
    expiry_date = _verify_date("TD1 expiry date", l2[8:14])
    expiry_check = _fix_digits(l2[14])
    nationality = _verify_letters("TD1 nationality", l2[15:18])
    optional2 = l2[18:29]
    _verify_document_number("TD1 document number", document_number, doc_check)
    _verify("TD1 birth date", birth_date, birth_check)
    _verify("TD1 expiry date", expiry_date, expiry_check)
    _verify(
        "TD1 composite",
        document_number
        + doc_check
        + optional1
        + birth_date
        + birth_check
        + expiry_date
        + expiry_check
        + optional2,
        l2[29],
    )
    surname, given_names = _parse_name(_fix_letters(l3))
    return MRZRecord(
        mrz_type="TD1",
        document_code=code,
        issuer=issuer,
        document_number=document_number.rstrip("<"),
        birth_date=birth_date,
        sex=_normalize_sex(l2[7]),
        expiry_date=expiry_date,
        nationality=nationality,
        surname=surname,
        given_names=given_names,
        optional_data1=optional1,
        optional_data2=optional2,
    )


def _parse_td2(l1: str, l2: str) -> MRZRecord:
    """Parse a TD2 (2x36) machine-readable zone."""
    code = _verify_letters("TD2 document code", l1[0:2])
    issuer = _verify_letters("TD2 issuer", l1[2:5])
    document_number = l2[0:9]
    doc_check = _fix_digits(l2[9])
    nationality = _verify_letters("TD2 nationality", l2[10:13])
    birth_date = _verify_date("TD2 birth date", l2[13:19])
    birth_check = _fix_digits(l2[19])
    expiry_date = _verify_date("TD2 expiry date", l2[21:27])
    expiry_check = _fix_digits(l2[27])
    optional_data = l2[28:35]
    _verify_document_number("TD2 document number", document_number, doc_check)
    _verify("TD2 birth date", birth_date, birth_check)
    _verify("TD2 expiry date", expiry_date, expiry_check)
    _verify(
        "TD2 composite",
        document_number
        + doc_check
        + birth_date
        + birth_check
        + expiry_date
        + expiry_check
        + optional_data,
        l2[35],
    )
    surname, given_names = _parse_name(_fix_letters(l1[5:36]))
    return MRZRecord(
        mrz_type="TD2",
        document_code=code,
        issuer=issuer,
        document_number=document_number.rstrip("<"),
        birth_date=birth_date,
        sex=_normalize_sex(l2[20]),
        expiry_date=expiry_date,
        nationality=nationality,
        surname=surname,
        given_names=given_names,
        optional_data1=optional_data,
    )


def _parse_td3(l1: str, l2: str) -> MRZRecord:
    """Parse a TD3 (2x44) machine-readable zone."""
    code = _verify_letters("TD3 document code", l1[0:2])
    issuer = _verify_letters("TD3 issuer", l1[2:5])
    document_number = l2[0:9]
    doc_check = _fix_digits(l2[9])
    nationality = _verify_letters("TD3 nationality", l2[10:13])
    birth_date = _verify_date("TD3 birth date", l2[13:19])
    birth_check = _fix_digits(l2[19])
    expiry_date = _verify_date("TD3 expiry date", l2[21:27])
    expiry_check = _fix_digits(l2[27])
    optional_data = l2[28:42]
    optional_check = _fix_digits(l2[42])
    _verify_document_number("TD3 document number", document_number, doc_check)
    _verify("TD3 birth date", birth_date, birth_check)
    _verify("TD3 expiry date", expiry_date, expiry_check)
    _verify("TD3 personal number", optional_data, optional_check)
    _verify(
        "TD3 composite",
        document_number
        + doc_check
        + birth_date
        + birth_check
        + expiry_date
        + expiry_check
        + optional_data
        + optional_check,
        l2[43],
    )
    surname, given_names = _parse_name(_fix_letters(l1[5:44]))
    return MRZRecord(
        mrz_type="TD3",
        document_code=code,
        issuer=issuer,
        document_number=document_number.rstrip("<"),
        birth_date=birth_date,
        sex=_normalize_sex(l2[20]),
        expiry_date=expiry_date,
        nationality=nationality,
        surname=surname,
        given_names=given_names,
        optional_data1=optional_data,
    )


class MRZReader:
    """Reads the ICAO 9303 machine-readable zone from document images.

    Args:
        reader: RapidOCRReader to reuse. Default-constructed when None.
        log_level: Logging level for the reader's logger.

    Raises:
        BlitzIDError: If the ``ocr`` extra (rapidocr) is not installed.
    """

    def __init__(
        self,
        reader: RapidOCRReader | None = None,
        log_level: int = logging.INFO,
    ) -> None:
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(log_level)
        self.reader = (
            reader if reader is not None else RapidOCRReader(log_level=log_level)
        )

    def read(self, image_input: ImageInput) -> MRZRecord:
        """Read the MRZ from an image.

        Args:
            image_input: Path, NumPy array (BGR), or PIL Image.

        Returns:
            The parsed :class:`MRZRecord`.

        Raises:
            ImageError: If the image cannot be loaded or is invalid.
            ModelError: If the OCR engine returns a partial result.
            MRZError: If no MRZ is found or a check digit fails.
        """
        return self.parse(self.reader.read(image_input))

    def parse(self, lines: list[OCRText]) -> MRZRecord:
        """Parse the MRZ from already-read OCR text lines.

        Lets callers that ran :class:`RapidOCRReader` once reuse its
        output instead of running the OCR pass a second time.

        Args:
            lines: Text lines as returned by :meth:`RapidOCRReader.read`.

        Returns:
            The parsed :class:`MRZRecord`.

        Raises:
            MRZError: If no MRZ is found or a check digit fails.
        """
        record = _read_mrz(lines)
        self.logger.info("MRZ read: %s", record.mrz_type)
        return record
