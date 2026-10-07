"""BarcodeReader — AAMVA PDF417 driver's-license reading.

US and Canadian driver's licenses carry the machine-readable record on
the back as a PDF417 barcode encoding an AAMVA DL/ID payload. The
decoder runs zxing-cpp (the ``barcode`` extra); the parser is pure
logic over the decoded text and mirrors the MRZ layer's posture: the
header's subfile offsets, lengths, and number-of-entries count are
validated, mandatory fields are gated, and a failure raises
:class:`~blitzid.exceptions.BarcodeError` naming the offending field —
nothing is invented or partially repaired.

Payload layout (AAMVA versions 02-14)::

    @ LF RS CR "ANSI " IIN(6) version(2) jurisdiction(2) count(2)
    <subfile designators: type(2) offset(4) length(4) each>
    <subfile data at the declared offsets, each ending in CR>

Subfile offsets are absolute from the first byte (the ``@``). Each
subfile holds 3-character element IDs followed by their values, one
per LF-separated line; the DL/ID/EN subfile carries the standard
elements (EN is the pre-2025 enhanced-license subfile) and
jurisdiction ``Z..`` subfiles carry state-specific extras.

Element semantics differ across editions, so each edition's tag map is
pinned separately: versions 02-03 encode the given name in DCT (first
and middle combined), versions 04-14 in DAC; the family name is DCS
throughout. DAA is the full-name element only in the version-01
(pre-2003) layout, which this module does not support — within the
02-14 range it is never mapped, and a jurisdiction that re-uses DAA
keeps the value raw in ``extra_tags``. Date elements
(DBB, DBD, DBA) are MMDDYYYY (U.S.) or CCYYMMDD (Canada) and normalize
via :func:`blitzid._dates.parse_date`; DBC sex encodes 1=male,
2=female, 9=unspecified, and any other code maps to unspecified with
the raw code kept in ``extra_tags``.
"""

from __future__ import annotations

import logging
from collections.abc import Mapping
from dataclasses import asdict, dataclass, replace
from datetime import date
from types import MappingProxyType
from typing import Any

from blitzid._dates import parse_date
from blitzid._image import ImageInput, load_image
from blitzid.exceptions import BarcodeError, BlitzIDError

__all__ = ["BarcodeReader", "BarcodeRecord"]

_MAGIC = "@\n\x1e\rANSI "
_HEADER_LEN = 21
_ENTRY_LEN = 10
_DATA_KINDS = frozenset({"DL", "ID", "EN"})
_VERSIONS = frozenset(f"{number:02d}" for number in range(2, 15))
_LEGACY_GIVEN = frozenset({"02", "03"})
_COMMON_MAPPED = frozenset({"DCS", "DAD", "DBA", "DBB", "DBC", "DBD", "DAQ", "DAU"})
_SEX = MappingProxyType({"1": "M", "2": "F", "9": "X"})


@dataclass(frozen=True)
class BarcodeRecord:
    """Parsed AAMVA PDF417 driver's-license record.

    Attributes:
        jurisdiction: Issuer identification number of the issuing
            jurisdiction.
        aamva_version: AAMVA version number, ``"02"`` through ``"14"``.
        jurisdiction_version: The jurisdiction's version number.
        family_name: Family name (DCS).
        given_name: Given name (DAC, or the first DCT part for versions
            02-03).
        middle_name: Middle name(s) (DAD, or the remaining DCT parts
            when DAD is absent), empty when none are encoded.
        birth_date: Date of birth (DBB).
        issue_date: Document issue date (DBD).
        expiry_date: Document expiration date (DBA).
        sex: ``"M"``, ``"F"``, or ``"X"`` (DBC 1/2/9); any other DBC
            code maps to ``"X"`` with the raw code kept in
            ``extra_tags``.
        license_number: Customer ID number (DAQ).
        height: Height with its unit, e.g. ``"069 in"`` (DAU), or None
            when absent.
        extra_tags: Raw values of every un-mapped element —
            jurisdiction ``Z..`` extras, standard-but-unmapped fields,
            and a non-standard DBC code alike.
        symbology: The decoded barcode symbology, stamped by the reader
            (``"PDF417"`` for the AAMVA payload), None for records
            built directly by the parser.
    """

    jurisdiction: str
    aamva_version: str
    jurisdiction_version: str
    family_name: str
    given_name: str
    middle_name: str
    birth_date: date
    issue_date: date
    expiry_date: date
    sex: str
    license_number: str
    height: str | None
    extra_tags: Mapping[str, str]
    symbology: str | None = None


@dataclass(frozen=True)
class _Subfile:
    """One directory entry plus its parsed elements."""

    kind: str
    elements: Mapping[str, str]


def _is_digits(value: str) -> bool:
    """Whether *value* is ASCII digits (isdigit alone admits Unicode)."""
    return value.isascii() and value.isdigit()


def _field(elements: Mapping[str, str], tag: str, label: str) -> str:
    """The mandatory element's value, erroring when it is missing."""
    value = _optional(elements, tag)
    if not value:
        raise BarcodeError(f"missing {label} ({tag})")
    return value


def _date_field(elements: Mapping[str, str], tag: str, label: str) -> date:
    """Parse a mandatory MMDDYYYY/CCYYMMDD date element."""
    raw = _field(elements, tag, label)
    parsed = parse_date(raw)
    if not isinstance(parsed, date):
        raise BarcodeError(f"invalid {label} ({tag}): {raw!r} is not a real date")
    return parsed


def _sex(elements: Mapping[str, str]) -> tuple[str, str | None]:
    """Map the DBC sex code to M/F/X.

    Returns ``(sex, raw)``: the mapped value and, for a non-standard
    DBC code, that raw code to keep in ``extra_tags``. A non-standard
    code maps to ``"X"`` (unspecified) instead of failing the record.
    """
    raw = _field(elements, "DBC", "sex")
    mapped = _SEX.get(raw)
    if mapped is None:
        return "X", raw
    return mapped, None


def _optional(elements: Mapping[str, str], tag: str) -> str | None:
    """An optional element's stripped value, None when absent."""
    value = elements.get(tag, "").strip()
    return value or None


def _names(version: str, elements: Mapping[str, str]) -> tuple[str, str, str]:
    """The edition-specific name split: (family, given, middle).

    Versions 02-03 hold the given name in DCT as ``FIRST,MIDDLE``;
    versions 04-14 split it into DAC (given) and DAD (middle). DAD
    wins over a DCT remainder for the middle name.
    """
    family = _field(elements, "DCS", "family name")
    middle = _optional(elements, "DAD") or ""
    if version in _LEGACY_GIVEN:
        parts = _field(elements, "DCT", "given name").split(",")
        given = parts[0]
        if not middle and len(parts) > 1:
            middle = ",".join(parts[1:])
    else:
        given = _field(elements, "DAC", "given name")
    return family, given, middle


def _parse_subfile(text: str, offset: int, length: int) -> Mapping[str, str]:
    """Split a subfile's body into its ``{tag: value}`` elements."""
    body = text[offset + 2 : offset + length - 1]
    elements: dict[str, str] = {}
    for line in body.split("\n"):
        if len(line) < 3:
            continue
        elements[line[:3]] = line[3:].rstrip("\r")
    return MappingProxyType(elements)


def _is_designator(value: str) -> bool:
    """Whether *value* is a two-letter ASCII subfile designator."""
    return value.isascii() and value.isalpha() and value.isupper()


def _parse_entry(text: str, index: int, directory_end: int) -> _Subfile:
    """Validate one subfile directory entry and its data region."""
    start = _HEADER_LEN + index * _ENTRY_LEN
    entry = text[start : start + _ENTRY_LEN]
    kind, offset, length = entry[0:2], entry[2:6], entry[6:10]
    if not (_is_designator(kind) and _is_digits(offset) and _is_digits(length)):
        raise BarcodeError(f"malformed subfile directory entry {index}")
    offset_i, length_i = int(offset), int(length)
    if length_i < 3:
        raise BarcodeError(f"invalid subfile {kind!r} length: {length_i}")
    if offset_i < directory_end:
        raise BarcodeError(f"subfile {kind!r} offset lies inside the header")
    if offset_i + length_i > len(text):
        raise BarcodeError(f"subfile {kind!r} exceeds the payload")
    if text[offset_i : offset_i + 2] != kind:
        raise BarcodeError(f"subfile {kind!r} offset does not match its data")
    if text[offset_i + length_i - 1] != "\r":
        raise BarcodeError(f"subfile {kind!r} is missing its segment terminator")
    return _Subfile(kind=kind, elements=_parse_subfile(text, offset_i, length_i))


def _parse_header(text: str) -> tuple[str, str, str, list[_Subfile]]:
    """Validate the compliance prefix, version block, and subfile
    directory, returning ``(iin, version, jurisdiction_version, subfiles)``.
    """
    if not text.startswith(_MAGIC):
        raise BarcodeError("not an AAMVA payload (missing '@\\n\\x1e\\rANSI ' header)")
    version = text[15:17]
    if version not in _VERSIONS:
        raise BarcodeError(f"unsupported AAMVA version: {version!r}")
    if not _is_digits(text[9:15]):
        raise BarcodeError(f"invalid issuer identification number: {text[9:15]!r}")
    if not _is_digits(text[17:19]):
        raise BarcodeError(f"invalid jurisdiction version: {text[17:19]!r}")
    if not _is_digits(text[19:21]):
        raise BarcodeError(f"invalid number of entries: {text[19:21]!r}")
    count = int(text[19:21])
    if count < 1:
        raise BarcodeError(f"invalid number of entries: {count}")
    directory_end = _HEADER_LEN + count * _ENTRY_LEN
    return (
        text[9:15],
        version,
        text[17:19],
        [_parse_entry(text, index, directory_end) for index in range(count)],
    )


def _mapped_ids(version: str) -> frozenset[str]:
    """The tag IDs one edition maps to fields (DCT vs DAC for the given name)."""
    return _COMMON_MAPPED | ({"DCT"} if version in _LEGACY_GIVEN else {"DAC"})


def _merge(
    subfiles: list[_Subfile], version: str
) -> tuple[Mapping[str, str], Mapping[str, str]]:
    """Separate the primary data subfile's elements from the raw extras.

    Returns ``(primary, extra_tags)``: the primary DL/ID/EN subfile
    drives the mapped fields; every element of every subfile minus the
    tags the edition actually maps stays raw in ``extra_tags``.
    """
    primary: Mapping[str, str] | None = None
    extra: dict[str, str] = {}
    for subfile in subfiles:
        if subfile.kind in _DATA_KINDS and primary is None:
            primary = subfile.elements
        extra.update(subfile.elements)
    if primary is None:
        raise BarcodeError("no DL/ID data subfile")
    for tag in _mapped_ids(version):
        extra.pop(tag, None)
    return primary, MappingProxyType(extra)


def _parse_aamva(text: str) -> BarcodeRecord:
    """Parse an AAMVA PDF417 payload.

    Args:
        text: The decoded payload, starting with the
            ``@\\n\\x1e\\rANSI `` header.

    Returns:
        The parsed :class:`BarcodeRecord`.

    Raises:
        BarcodeError: If the header, subfile directory, or a mandatory
            field is missing or malformed.
    """
    jurisdiction, version, jurisdiction_version, subfiles = _parse_header(text)
    primary, extra = _merge(subfiles, version)
    family_name, given_name, middle_name = _names(version, primary)
    sex, raw_sex = _sex(primary)
    if raw_sex is not None:
        extra = MappingProxyType({**extra, "DBC": raw_sex})
    return BarcodeRecord(
        jurisdiction=jurisdiction,
        aamva_version=version,
        jurisdiction_version=jurisdiction_version,
        family_name=family_name,
        given_name=given_name,
        middle_name=middle_name,
        birth_date=_date_field(primary, "DBB", "birth date"),
        issue_date=_date_field(primary, "DBD", "issue date"),
        expiry_date=_date_field(primary, "DBA", "expiry date"),
        sex=sex,
        license_number=_field(primary, "DAQ", "license number"),
        height=_optional(primary, "DAU"),
        extra_tags=extra,
    )


def _record_dict(record: BarcodeRecord) -> dict[str, Any]:
    """The JSON-safe section body for one :class:`BarcodeRecord`.

    ``asdict`` mirrors the record's fields so a new field appears in
    the API automatically, but it deep-copies every value and cannot
    copy the frozen ``extra_tags`` mapping, so the record is rebuilt
    with a plain dict first. Only the JSON-unfriendly dates are
    converted here.
    """
    body = asdict(replace(record, extra_tags=dict(record.extra_tags)))
    body["birth_date"] = record.birth_date.isoformat()
    body["issue_date"] = record.issue_date.isoformat()
    body["expiry_date"] = record.expiry_date.isoformat()
    return body


def _load_zxingcpp() -> Any:
    """Import zxing-cpp, mapping a missing ``barcode`` extra to BlitzIDError."""
    try:
        import zxingcpp
    except ImportError as e:
        raise BlitzIDError(
            "zxing-cpp is not installed. "
            "Install the barcode extra: pip install blitzid[barcode]"
        ) from e
    return zxingcpp


class BarcodeReader:
    """Reads the AAMVA PDF417 barcode from a driver's-license back.

    Args:
        log_level: Logging level for the reader's logger.

    Raises:
        BlitzIDError: If the ``barcode`` extra (zxing-cpp) is not installed.
    """

    def __init__(self, log_level: int = logging.INFO) -> None:
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(log_level)
        self._zxing = _load_zxingcpp()

    def read(self, image_input: ImageInput) -> BarcodeRecord:
        """Decode and parse the PDF417 barcode from an image.

        Every decoded barcode is tried and the first payload that
        parses as AAMVA wins, so one corrupt candidate beside a valid
        code does not fail the read.

        Args:
            image_input: Path, NumPy array (BGR), or PIL Image.

        Returns:
            The parsed :class:`BarcodeRecord`.

        Raises:
            ImageError: If the image cannot be loaded or is invalid.
            BarcodeError: If the image decodes no barcode at all, no
                decoded payload is an AAMVA payload, or every AAMVA
                payload fails validation.
        """
        img = load_image(image_input, self.logger)
        barcodes = self._zxing.read_barcodes(img)
        if not barcodes:
            raise BarcodeError("no PDF417 barcode found in the image")
        return self._first_record(barcodes)

    def _first_record(self, barcodes: Any) -> BarcodeRecord:
        """The first candidate barcode that parses as an AAMVA payload.

        A candidate carrying the AAMVA header but failing validation
        does not stop the scan of the remaining barcodes; when none
        parses, the first such validation error is raised, or
        :class:`BarcodeError` when no barcode carried the header.
        """
        error: BarcodeError | None = None
        for barcode in barcodes:
            text = barcode.bytes.decode("latin-1")
            if not text.startswith(_MAGIC):
                continue
            try:
                record = _parse_aamva(text)
            except BarcodeError as e:
                error = error or e
                continue
            record = replace(record, symbology=str(barcode.format))
            self.logger.info(
                "barcode read: version %s, jurisdiction %s",
                record.aamva_version,
                record.jurisdiction,
            )
            return record
        if error is not None:
            raise error
        raise BarcodeError(
            "no AAMVA payload among the decoded barcodes "
            "(missing '@\\n\\x1e\\rANSI ' header)"
        )
