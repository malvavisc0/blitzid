"""MRZReader — ICAO 9303 machine-readable zone reading.

Selects MRZ lines from OCR text, validates them via ICAO 9303 check
digits plus the letter-only fields (document code, issuer, nationality),
and parses TD1 (3x30, ID cards), TD2 (2x36), and TD3 (2x44, passports)
layouts into :class:`MRZRecord` fields. Runs on top of the ``ocr``
extra's RapidOCRReader. No fuzzy OCR-error correction — a zone that
does not validate raises :class:`~blitzid.exceptions.MRZError` with the
failing field.
"""

from __future__ import annotations

import logging
from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from datetime import datetime
from itertools import combinations

from .._image import ImageInput
from ..exceptions import MRZError
from .ocr import OCRText, RapidOCRReader

__all__ = ["MRZReader", "MRZRecord"]

_CHARSET = frozenset("ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789<")
_LETTERS = frozenset("ABCDEFGHIJKLMNOPQRSTUVWXYZ")
_WEIGHTS = (7, 3, 1)
_LINE_COUNT = {30: 3, 36: 2, 44: 2}


@dataclass(frozen=True)
class MRZRecord:
    """Parsed ICAO 9303 machine-readable zone.

    Attributes:
        mrz_type: "TD1", "TD2", or "TD3".
        document_code: Two-character document code, e.g. "P<" or "I<".
        issuer: Three-character issuing state code.
        document_number: Document number, fillers stripped.
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
    computed = _check_digit(text)
    if not expected.isdigit() or computed != int(expected):
        raise MRZError(
            f"{label} check digit mismatch: printed {expected!r}, computed {computed}"
        )


def _verify_letters(label: str, code: str) -> None:
    """Verify a letters-only MRZ field (trailing fillers tolerated).

    Catches OCR confusions such as O→0 or I→| in fields that carry no
    check digit (document code, issuer, nationality). Trailing '<'
    fillers are ignored, so "UTO" and "UT<" both pass; the field must
    still contain at least one letter.

    Raises:
        MRZError: If *code* contains a non-letter character.
    """
    letters = code.rstrip("<")
    if not letters or not all(ch in _LETTERS for ch in letters):
        raise MRZError(f"invalid {label}: {code!r}")


def _verify_date(label: str, yymmdd: str) -> None:
    """Verify a YYMMDD MRZ date is a real calendar date.

    Raises:
        MRZError: If the field is not a real calendar date.
    """
    try:
        datetime.strptime(yymmdd, "%y%m%d")
    except ValueError as e:
        raise MRZError(f"invalid {label}: {yymmdd!r} is not a real date") from e


def _fillers_to_spaces(text: str) -> str:
    """Turn '<' fillers into single spaces, dropping empty parts."""
    return " ".join(part for part in text.split("<") if part)


def _parse_name(field: str) -> tuple[str, str]:
    """Split a name field into ``(surname, given_names)``."""
    surname, _, given = field.partition("<<")
    return _fillers_to_spaces(surname), _fillers_to_spaces(given)


def _normalize_sex(sex: str) -> str:
    """Map an MRZ sex character to "M", "F", or "X".

    Raises:
        MRZError: If the character is not M, F, or "<".
    """
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
    candidates = sorted(
        (text.bbox[1], text.text.replace(" ", "").upper())
        for text in texts
        if _is_mrz_line(text.text.replace(" ", "").upper())
    )
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
    _verify("TD1 document number", l1[5:14], l1[14])
    _verify("TD1 birth date", l2[0:6], l2[6])
    _verify("TD1 expiry date", l2[8:14], l2[14])
    _verify("TD1 composite", l1[5:30] + l2[0:7] + l2[8:15] + l2[18:29], l2[29])
    _verify_date("TD1 birth date", l2[0:6])
    _verify_date("TD1 expiry date", l2[8:14])
    _verify_letters("TD1 document code", l1[0:2])
    _verify_letters("TD1 issuer", l1[2:5])
    _verify_letters("TD1 nationality", l2[15:18])
    surname, given_names = _parse_name(l3)
    return MRZRecord(
        mrz_type="TD1",
        document_code=l1[0:2],
        issuer=l1[2:5],
        document_number=l1[5:14].rstrip("<"),
        birth_date=l2[0:6],
        sex=_normalize_sex(l2[7]),
        expiry_date=l2[8:14],
        nationality=l2[15:18],
        surname=surname,
        given_names=given_names,
        optional_data1=l1[15:30],
        optional_data2=l2[18:29],
    )


def _parse_td2(l1: str, l2: str) -> MRZRecord:
    """Parse a TD2 (2x36) machine-readable zone."""
    _verify("TD2 document number", l2[0:9], l2[9])
    _verify("TD2 birth date", l2[13:19], l2[19])
    _verify("TD2 expiry date", l2[21:27], l2[27])
    _verify("TD2 composite", l2[0:10] + l2[13:20] + l2[21:35], l2[35])
    _verify_date("TD2 birth date", l2[13:19])
    _verify_date("TD2 expiry date", l2[21:27])
    _verify_letters("TD2 document code", l1[0:2])
    _verify_letters("TD2 issuer", l1[2:5])
    _verify_letters("TD2 nationality", l2[10:13])
    surname, given_names = _parse_name(l1[5:36])
    return MRZRecord(
        mrz_type="TD2",
        document_code=l1[0:2],
        issuer=l1[2:5],
        document_number=l2[0:9].rstrip("<"),
        birth_date=l2[13:19],
        sex=_normalize_sex(l2[20]),
        expiry_date=l2[21:27],
        nationality=l2[10:13],
        surname=surname,
        given_names=given_names,
        optional_data1=l2[28:35],
    )


def _parse_td3(l1: str, l2: str) -> MRZRecord:
    """Parse a TD3 (2x44) machine-readable zone."""
    _verify("TD3 document number", l2[0:9], l2[9])
    _verify("TD3 birth date", l2[13:19], l2[19])
    _verify("TD3 expiry date", l2[21:27], l2[27])
    _verify("TD3 personal number", l2[28:42], l2[42])
    _verify("TD3 composite", l2[0:10] + l2[13:20] + l2[21:43], l2[43])
    _verify_date("TD3 birth date", l2[13:19])
    _verify_date("TD3 expiry date", l2[21:27])
    _verify_letters("TD3 document code", l1[0:2])
    _verify_letters("TD3 issuer", l1[2:5])
    _verify_letters("TD3 nationality", l2[10:13])
    surname, given_names = _parse_name(l1[5:44])
    return MRZRecord(
        mrz_type="TD3",
        document_code=l1[0:2],
        issuer=l1[2:5],
        document_number=l2[0:9].rstrip("<"),
        birth_date=l2[13:19],
        sex=_normalize_sex(l2[20]),
        expiry_date=l2[21:27],
        nationality=l2[10:13],
        surname=surname,
        given_names=given_names,
        optional_data1=l2[28:42],
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
        lines = self.reader.read(image_input)
        record = _read_mrz(lines)
        self.logger.info("MRZ read: %s", record.mrz_type)
        return record
